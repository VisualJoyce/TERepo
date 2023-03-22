from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import BertModel, PreTrainedModel
from transformers.models.bert.modeling_bert import CrossEntropyLoss, BertEncoder
from transformers.utils import ModelOutput

from terepo.models.tagging.hct.configuration_hct import HCTConfig

cc = SmoothingFunction()


def convert_tokens_to_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    return ' '.join(tokens).replace(' ##', '').strip()


def filter_spans(starts, ends, max_i, stop_i=0):
    for i, start in enumerate(starts):
        end = ends[i]
        if start == stop_i or start > end or start >= max_i:
            starts[i] = ends[i] = -1
            continue
    starts = [s for s in starts if s > -1]
    ends = [e for e in ends if e > -1]
    assert (len(starts) == len(ends))
    return starts, ends


def get_sp_strs(start_lst, end_lst, context_len):
    max_i = context_len - 1
    starts, ends = filter_spans(start_lst, end_lst, max_i)
    if not starts:
        starts.append(0)
        ends.append(0)
    starts, ends = dutils.ilst2str(starts), dutils.ilst2str(ends)
    return starts, ends


def load_rules(rule_path, mask='_', fmask='{}'):
    with open(rule_path, encoding='utf8') as f:
        rules = [''] + [l.strip().replace(mask, fmask) for l in f]
    rule_slot_cnts = [sum(int(y == fmask) for y in x.split()) for x in rules]
    return rules, rule_slot_cnts


def get_config(params, bert_class, bleu_rl):
    config = BertConfig.from_pretrained(bert_class)
    config.bert_class = bert_class
    config.device = params.device
    config.rl_model = 'bleu' if bleu_rl else None
    config.rl_ratio = params.rl_ratio

    config.num_labels = len(params.tag2idx)
    config.tags = params.idx2tag
    config.pad_tag_id = params.pad_tag_id

    config.rules = params.rules
    config.rule_slot_cnts = params.rule_slot_cnts
    config.max_sp_len = params.max_sp_len
    config.additional_special_tokens = tuple(f'[SLOT{x}]' for x in range(params.max_sp_len))
    config.vocab_size += len(config.additional_special_tokens)
    return config


def safe_log(inp, eps=1e-45):
    return (inp + eps).log()


def sample_helper(logits, dim=2):
    samples = Categorical(logits=logits).sample()
    samples_prob = torch.gather(logits, dim, samples.unsqueeze(dim))
    return samples, samples_prob


def cls_loss(dist, refs, masks):
    refs = F.one_hot(refs, dist.shape[-1])
    loss = torch.sum(safe_log(dist) * refs.float(), dim=-1)
    num_tokens = torch.sum(masks).item()
    return -torch.sum(loss * masks) / num_tokens


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions, max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


class AdditiveAttention(nn.Module):
    def __init__(self, model_dim):
        super(AdditiveAttention, self).__init__()
        self.linear_concat = nn.Linear(model_dim * 2, model_dim)
        self.linear_logit = nn.Linear(model_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_concat.weight, std=.02)
        nn.init.normal_(self.linear_logit.weight, std=.02)
        nn.init.constant_(self.linear_concat.bias, 0.)
        nn.init.constant_(self.linear_logit.bias, 0.)

    def forward(self, queries, keys, values, mask):
        """ Additive attention mechanism. This layer is implemented using a
            one layer feed forward neural network
        :param queries: A tensor with shape [batch, heads, length_q, depth_k]
        :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
        :param values: A tensor with shape [batch, heads, length_kv, depth_v]
        :param bias: A tensor
        :param concat: A boolean value. If ``concat'' is set to True, then
            the computation of attention mechanism is following $tanh(W[q, k])$.
            When ``concat'' is set to False, the computation is following
            $tanh(Wq + Vk)$
        :param keep_prob: a scalar in [0, 1]
        :param dtype: An optional instance of tf.DType
        :param scope: An optional string, the scope of this layer
        :returns: A dict with the following keys:
            weights: A tensor with shape [batch, length_q, length_kv]
        """
        queries = queries.unsqueeze(dim=2)  # [bs, len_q, 1, size]
        keys = keys.unsqueeze(dim=1)  # [bs, 1, len_k, size]
        q = queries.expand(-1, -1, keys.shape[2], -1)
        k = keys.expand(-1, queries.shape[1], -1, -1)
        combined = torch.tanh(self.linear_concat(torch.cat((q, k), dim=-1)))
        logits = self.linear_logit(combined).squeeze(-1)  # [bs, len_q, len_k]
        if mask is not None:
            mask = torch.logical_not(mask)
            mask.masked_fill_(mask.all(-1, keepdim=True), 0)  # prevent NaN
        logits.masked_fill_(mask, -float('inf'))
        weights = nn.functional.softmax(logits, dim=-1)
        return weights


class SingleHeadedAttention(nn.Module):
    """Applies linear projection over concatenated queries and keys instead of
    dot-product.
    Args:
       model_dim (int): query/key/value hidden size
       dropout (float): proportion of weights dropped out
    """

    def __init__(self, model_dim, max_relative_positions=0):
        super(SingleHeadedAttention, self).__init__()
        self.model_dim = model_dim
        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions
        self.additive_attention = AdditiveAttention(model_dim)

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.model_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_keys.weight, std=.02)
        nn.init.normal_(self.linear_values.weight, std=.02)
        nn.init.normal_(self.linear_query.weight, std=.02)
        nn.init.normal_(self.final_linear.weight, std=.02)
        nn.init.constant_(self.linear_keys.bias, 0.)
        nn.init.constant_(self.linear_values.bias, 0.)
        nn.init.constant_(self.linear_query.bias, 0.)
        nn.init.constant_(self.final_linear.bias, 0.)

    def forward(self, key, query, value=None, mask=None, type=None):
        """Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """
        key = self.linear_keys(key)
        value = self.linear_values(key if value is None else value)
        query = self.linear_query(query)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            # 1 or key_len x key_len x model_dim
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        out_shape = (key.shape[0], -1, self.model_dim)
        query = query.view(*out_shape) / math.sqrt(self.model_dim)
        return self.additive_attention(query, key.view(*out_shape), value.view(*out_shape), mask)


class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position=0., dropout=0.1):
        super(SpanClassifier, self).__init__()
        self.span_st_attn = SingleHeadedAttention(hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = SingleHeadedAttention(hidden_dim, max_relative_positions=max_relative_position)
        self.src_rule_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.sp_emb = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = dropout
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def reset_parameters(self):
        nn.init.normal_(self.sp_emb.weight, std=2e-2)
        nn.init.constant_(self.sp_emb.bias, 0.)
        nn.init.normal_(self.src_rule_proj.weight, std=2e-2)
        nn.init.constant_(self.src_rule_proj.bias, 0.)

    def forward(self, hid, sp_width, max_sp_len, attention_mask, src_hid, src_mask, rule_emb):
        amask = torch.logical_and(src_mask.unsqueeze(2), attention_mask.unsqueeze(1)).float()
        attn_d = amask.sum(-1, keepdim=True)
        attn_d.masked_fill_(attn_d == 0, 1.)
        attn_w0 = amask / attn_d

        sts, eds, masks = [attn_w0], [attn_w0], []
        src_hid = F.relu(self.src_rule_proj(torch.cat((src_hid, rule_emb), 2)), inplace=True)
        hid1, hid2 = src_hid, src_hid
        for i in range(max_sp_len):
            mask = (i < sp_width).float()
            masks.append(mask)
            mask = torch.logical_and(mask.unsqueeze(-1), amask)
            hid1 = self.upd_hid(sts[-1], hid, hid1)
            hid2 = self.upd_hid(eds[-1], hid, hid2)
            sts.append(self.span_st_attn(hid, hid1, mask=mask))
            eds.append(self.span_ed_attn(hid, hid2, mask=mask))
        return torch.stack(sts[1:], -2), torch.stack(eds[1:], -2), torch.stack(masks, -1)

    @staticmethod
    def span_loss(start_dist, end_dist, start_positions, end_positions, seq_masks):
        if not seq_masks.any():
            return 0.
        span_st_loss = cls_loss(start_dist, start_positions, seq_masks)
        span_ed_loss = cls_loss(end_dist, end_positions, seq_masks)
        return span_st_loss + span_ed_loss

    def upd_hid(self, attn_w, hid, hidp):
        hidc = (hid.unsqueeze(1) * attn_w.unsqueeze(-1)).sum(2)
        hidc = F.relu(self.sp_emb(torch.cat((hidp, hidc), 2)), inplace=True)
        hidc = F.dropout(hidc, p=self.dropout, training=self.training)
        return hidc


class HCTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HCTConfig
    base_model_prefix = "hct"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class HCTOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_d: Optional[torch.FloatTensor] = None
    loss_labels: Optional[torch.FloatTensor] = None
    logits_d: torch.FloatTensor = None
    logits_labels: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HCTForSequenceTagging(HCTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rule_classifier = nn.Linear(config.hidden_size, config.num_rules)
        self.span_classifier = SpanClassifier(config.hidden_size)

        # self.rl_model = config.rl_model
        self.rl_ratio = config.rl_ratio
        self.eps = 1e-45
        self.bleu_fn = partial(sentence_bleu, weights=(.25,) * 4,
                               smoothing_function=cc.method3)
        self.loss_fn = CrossEntropyLoss()

    def apply_rl(self, logits, rule_logits, start_dist, end_dist, start_outputs,
                 end_outputs, src_idx, ctx_idx, attention_mask, act_loss_mask,
                 sp_loss_mask, input_ref, max_sp_len, perm=(0, 2, 1, 3)):
        samples_action, samples_action_prob, greedy_action = self.get_sample_greedy(logits)
        samples_rule, samples_rule_prob, greedy_rule = self.get_sample_greedy(rule_logits)

        bsz, seq_len, _, full_len = start_dist.shape
        start_dist, samples_start, samples_start_prob = self.sample_sp(start_dist, perm, seq_len, full_len)
        end_dist, samples_end, samples_end_prob = self.sample_sp(end_dist, perm, seq_len, full_len)

        nview = (bsz, max_sp_len, seq_len)
        samples_start, samples_end, samples_start_prob, samples_end_prob = \
            map(lambda x: self.reshape_sp(x, nview, perm[:-1]),
                (samples_start, samples_end, samples_start_prob, samples_end_prob))

        min_r, max_r = 1., -1.
        rewards = []
        act_loss_mask = act_loss_mask.float()
        for i in range(len(samples_start)):
            src_len, src_tokens = self.get_len_tokens(act_loss_mask[i], src_idx[i])
            ctx_len, ctx_tokens = self.get_len_tokens(attention_mask[i], ctx_idx[i])

            sample_str = self.decode_into_string(src_tokens, samples_action[i].tolist(), samples_rule[i].tolist(),
                                                 samples_start[i].tolist(), samples_end[i].tolist(), src_len,
                                                 context=ctx_tokens, context_len=ctx_len)
            greedy_str = self.decode_into_string(src_tokens, greedy_action[i].tolist(), greedy_rule[i].tolist(),
                                                 start_outputs[i].tolist(), end_outputs[i].tolist(), src_len,
                                                 context=ctx_tokens, context_len=ctx_len)

            input_ref_lst = [input_ref[i].split()]
            sample_score = self.bleu_fn(input_ref_lst, sample_str.split())
            greedy_score = self.bleu_fn(input_ref_lst, greedy_str.split())
            rewards.append(sample_score - greedy_score)
            min_r = min(min_r, rewards[-1])
            max_r = max(max_r, rewards[-1])

        rewards = torch.as_tensor(rewards, device=logits.device).unsqueeze(1)
        rewards_ = (rewards - min_r) / (max_r - min_r)
        rewards = torch.where(rewards_.isnan(), rewards, rewards_)
        loss_action_rl = self.rl_loss(samples_action_prob.squeeze(2), act_loss_mask, rewards)
        loss_rule_rl = self.rl_loss(samples_rule_prob.squeeze(2), act_loss_mask, rewards)

        rewards = rewards.unsqueeze(2)
        loss_st_rl = self.rl_loss(samples_start_prob, sp_loss_mask, rewards)
        loss_ed_rl = self.rl_loss(samples_end_prob, sp_loss_mask, rewards)

        loss_rl = loss_action_rl + loss_rule_rl + loss_st_rl + loss_ed_rl
        return loss_rl

    def get_active_loss(self, loss_mask, logits, labels, num_labels):
        active_loss = loss_mask.view(-1)
        active_logits = logits.view(-1, num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        return self.loss_fn(active_logits, active_labels)

    @staticmethod
    def get_sample_greedy(logits):
        samples, samples_prob = sample_helper(logits)
        greedy = logits.argmax(dim=-1)
        return samples, samples_prob, greedy

    @staticmethod
    def sample_sp(logits, perm, seq_len, full_len):
        logits = logits.permute(*perm).reshape(-1, seq_len, full_len)
        samples, samples_prob = sample_helper(logits)
        return logits, samples, samples_prob

    @staticmethod
    def reshape_sp(inp, nview, perm):
        return inp.view(*nview).permute(*perm)

    @staticmethod
    def gather_embed(inp_emb, inp_idx, inp_mask, dim=1):
        expand_dims = (-1,) * (dim + 1)
        ret = inp_emb.gather(dim, inp_idx.unsqueeze(2).expand(*expand_dims,
                                                              inp_emb.shape[-1])) * inp_mask.unsqueeze(2).to(
            inp_emb.dtype)
        return ret

    def rl_loss(self, prob, mask, rewards):
        loss = -F.log_softmax(prob + safe_log(mask), -1)
        loss *= rewards * mask
        msum = mask.sum()
        return loss.sum() / torch.clamp(mask.sum(), min=self.eps)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            source_index: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            span_width: Optional[torch.Tensor] = None,
            labels_action: Optional[torch.Tensor] = None,
            labels_rule: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], HCTOutput]:
        # def forward(self, input_data, labels_action, labels_start, labels_end,
        #             sp_width, rule, src_idx):
        #     input_ids, input_ref, attention_mask, seq_output, src_mask, src_output, \
        #     src_idx = self.unpack_data(input_data, src_idx)
        #     input_ids, input_ref = input_data
        seq_output = self.bert(input_ids, attention_mask=attention_mask)[0]

        source_mask = source_index.ne(self.pad_token_id)
        source_output = self.gather_embed(seq_output, source_index, source_mask)
        source_index = input_ids.gather(1, source_index) * source_mask.long()

        # logits, act_loss_mask, loss_action, rule_logits, loss_rule, rule_emb = \
        #     self.pred_tags(src_mask, src_output, src_idx, labels_action, rule)
        logits = self.classifier(source_output)
        rule_logits = self.rule_classifier(source_output)
        if not self.training:
            labels_rule = rule_logits.argmax(2)
        rule_emb = self.rule_embeds[labels_rule]

        max_sp_len = start_positions.shape[-1]
        start_dist, end_dist, sp_loss_mask = self.span_classifier(seq_output,
                                                                  span_width, max_sp_len, attention_mask, source_output,
                                                                  source_mask, rule_emb)
        start_outputs = start_dist.argmax(dim=-1)
        end_outputs = end_dist.argmax(dim=-1)

        if self.rl_model is not None and self.training:
            act_loss_mask = labels_action.ne(self.pad_tag_id)
            loss_action = self.get_active_loss(act_loss_mask, logits, labels_action,
                                               self.num_labels)

            loss_rule = self.get_active_loss(act_loss_mask, rule_logits, labels_rule,
                                             self.num_rules)

            sp_loss_mask = torch.logical_and(labels_rule.gt(0).unsqueeze(2), sp_loss_mask).float()
            loss_span = self.span_classifier.span_loss(start_dist, end_dist,
                                                       start_positions, end_positions, sp_loss_mask)

            loss = loss_action + loss_rule + loss_span

            loss_rl = self.apply_rl(logits, rule_logits, start_dist, end_dist,
                                    start_outputs, end_outputs, source_index, input_ids, attention_mask,
                                    act_loss_mask, sp_loss_mask, input_ref, max_sp_len)
            loss = (1. - self.rl_ratio) * loss + self.rl_ratio * loss_rl

        # outputs = (loss, logits, rule_logits, start_outputs, end_outputs)
        return HCTOutput(
            loss=loss,
            logits=logits,
        )
