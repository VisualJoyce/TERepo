from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from allennlp.modules import TimeDistributed
from transformers import PreTrainedModel, AutoModel, RobertaModel
from transformers.models.bert import BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.utils import ModelOutput

from terepo.models import register_model
from terepo.optim.loss import FocalLoss
from terepo.utils.misc import mismatched_embeddings, logits_mask, NoOp, MatchingLayer
from .configuration_gector import GECToRConfig


class GECToRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GECToRConfig
    base_model_prefix = "gector"
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
class GECToROutput(ModelOutput):
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


@register_model("gector")
class GECToRModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode

        self.bert = BertModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))
        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)

        self.label_smoothing = 0.1

        self.loss_labels = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.loss_tags = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)

        logits_d = self.tag_detect_projection_layer(orig_embeddings)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(orig_embeddings))

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            # loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, original_mask,
            #                                                  label_smoothing=self.label_smoothing)
            # loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, original_mask)
            loss_labels = self.loss_labels(logits_labels.view(-1, self.config.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model("gector_focal")
class GECToRFocalModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode

        self.transformer = AutoModel.from_config(config.encoder)
        # self.transformer = RobertaModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))
        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)

        self.loss_labels = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.loss_tags = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)

        logits_d = self.tag_detect_projection_layer(orig_embeddings)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(orig_embeddings))

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            loss_labels = self.loss_labels(logits_labels.view(-1, self.config.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model("gector_focal_label_embedding")
class GECToRFocalLabelEmbeddingModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode
        self.label_vocab_size = config.label_vocab_size

        self.bert = BertModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))

        self.register_buffer('label_candidates', torch.arange(config.label_vocab_size))
        self.label_embeddings = nn.Embedding(config.label_vocab_size, config.hidden_size)

        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        # self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)

        self.loss_labels = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.loss_tags = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.init_weights()

    def resize_label_embeddings(self, label_vocab_size: int) -> nn.Embedding:
        old_embeddings = self.label_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, label_vocab_size)
        # self._resize_final_logits_bias(new_num_tokens)
        self.label_embeddings = new_embeddings
        self.label_vocab_size = label_vocab_size
        self.register_buffer('label_candidates', torch.arange(label_vocab_size))
        return new_embeddings

    def vocab(self, blank_states):
        label_embeddings = self.label_embeddings(self.label_candidates)
        return torch.einsum('bld,nd->bln', [blank_states, label_embeddings])  # (b, 256, 10)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)

        logits_d = self.tag_detect_projection_layer(orig_embeddings)
        # logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(orig_embeddings))
        logits_labels = self.vocab(self.predictor_dropout(orig_embeddings))

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            loss_labels = self.loss_labels(logits_labels.view(-1, self.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HopLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Dense
        self._query_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self._key_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self._layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor = None):
        query = self._query_embedding(inputs)
        key = self._key_embedding(inputs)

        scores = torch.matmul(query, key.permute(0, 2, 1))
        scores = scores * (1 / (query.shape[-1] ** (1 / 2)))
        scores = logits_mask(scores, masks)
        attention_probs = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attention_probs, inputs)
        return self._layernorm(inputs + context)


@register_model("gector_focal_detach")
class GECToRFocalDetachModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode

        self.bert = BertModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))
        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)

        self.hop_layer = HopLayer(config)

        self.loss_labels = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.loss_tags = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(orig_embeddings))

        state = self.hop_layer(orig_embeddings, original_mask[:, None, :])
        logits_d = self.tag_detect_projection_layer(state)

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            loss_labels = self.loss_labels(logits_labels.view(-1, self.config.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model("gector_focal_matching")
class GECToRFocalMatchingModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode

        self.bert = BertModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))
        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)

        self.matching_layer = MatchingLayer(config)

        self.loss_labels = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.loss_tags = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(orig_embeddings))

        # state = self.matching_layer(self.predictor_dropout(orig_embeddings), original_mask[:, None, :])
        state = self.matching_layer(orig_embeddings, original_mask[:, None, :])
        logits_d = self.tag_detect_projection_layer(state)

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            loss_labels = self.loss_labels(logits_labels.view(-1, self.config.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model("gector_focal_label_embedding_matching")
class GECToRFocalLabelEmbeddingMatchingModel(GECToRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sub_token_mode = config.sub_token_mode
        self.label_vocab_size = config.label_vocab_size

        self.bert = BertModel(config.encoder)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))

        self.register_buffer('label_candidates', torch.arange(config.label_vocab_size))
        self.label_embeddings = nn.Embedding(config.label_vocab_size, config.hidden_size)

        self.tag_detect_projection_layer = torch.nn.Linear(config.hidden_size, config.detect_vocab_size)
        # self.tag_labels_projection_layer = torch.nn.Linear(config.hidden_size, config.label_vocab_size)
        self.matching_layer = MatchingLayer(config)

        self.loss_labels = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.loss_tags = FocalLoss(gamma=config.focal_gamma, reduction='mean')
        self.init_weights()

    def resize_label_embeddings(self, label_vocab_size: int) -> nn.Embedding:
        old_embeddings = self.label_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, label_vocab_size)
        # self._resize_final_logits_bias(new_num_tokens)
        self.label_embeddings = new_embeddings
        self.label_vocab_size = label_vocab_size
        self.register_buffer('label_candidates', torch.arange(label_vocab_size))
        return new_embeddings

    def vocab(self, blank_states):
        label_embeddings = self.label_embeddings(self.label_candidates)
        return torch.einsum('bld,nd->bln', [blank_states, label_embeddings])  # (b, 256, 10)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            offsets: Optional[torch.Tensor] = None,
            original_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            d_tags: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        sequence_output = outputs[0]

        orig_embeddings = mismatched_embeddings(sequence_output, offsets, self.sub_token_mode)

        state = self.matching_layer(orig_embeddings, original_mask[:, None, :])
        logits_d = self.tag_detect_projection_layer(state)
        logits_labels = self.vocab(self.predictor_dropout(orig_embeddings))

        total_loss = loss_d = loss_labels = None
        if labels is not None and d_tags is not None:
            loss_labels = self.loss_labels(logits_labels.view(-1, self.label_vocab_size),
                                           labels.view(-1))
            loss_d = self.loss_tags(logits_d.view(-1, self.config.detect_vocab_size), d_tags.view(-1))
            total_loss = loss_labels + loss_d

        if not return_dict:
            output = (logits_d, logits_labels) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GECToROutput(
            loss=total_loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
