from glob import glob

import webdataset as wds
from transformers import BertTokenizer

# sys.path.append('/home/vimos/Data/Text/chn_nlp/TextEditing/WenXin/src')
data_dir = "/home/vimos/git/WenXin/data/annotations/text_editing/en/canard/test"
dataset = wds.WebDataset(glob(f"{data_dir}/*")).decode()
tokenizer = BertTokenizer.from_pretrained("/home/vimos/Data/Text/pretrained_models/text_editing/hct-canard/")


def load_rules(rule_path, mask='_', fmask='{}'):
    with open(rule_path, encoding='utf8') as f:
        rules = [''] + [l.strip().replace(mask, fmask) for l in f]
    rule_slot_cnts = [sum(int(y == fmask) for y in x.split()) for x in rules]
    return rules, rule_slot_cnts


rules_vocab, _ = load_rules(
    "/home/vimos/Data/Text/TextEditing/UtteranceRewriting/RaST_data/canard/train/rule_affinity.txt")
to_int = lambda x: int(x) + 1
idx2tag = ["KEEP", "DELETE"]
tag2idx = {tag: idx for idx, tag in enumerate(idx2tag)}


def upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr):
    rem = sw = seq_width[ptr]
    cur_len, cur_rule = curr_len_list[ptr], rule_seq[ptr]
    return rem, sw, cur_len, cur_rule, ptr + 1


def _split_multi_span(seq):
    sid = 0
    seq_out = [sid]
    seq_width = [1]
    for si, i in enumerate(seq):
        if ',' in i:
            slst = list(map(to_int, i.split(',')))[:3]
            seq_out.extend(slst)
            seq_width.append(len(slst))
        else:
            seq_out.append(to_int(i))
            seq_width.append(1)
    return seq_out, seq_width


def _split_to_wordpieces_span(tokens, label_action):
    bert_tokens = []
    bert_label_action = []
    source_indices = []
    cum_num_list = []
    curr_len_list = []
    cum_num = 0
    src_start = orig_start = len(tokens)
    for i, token in enumerate(tokens):
        pieces = tokenizer.tokenize(token)
        if token == '|':
            src_start = len(bert_tokens) + 1
            orig_start = i + 1

        bert_label_action.extend([label_action[i]] * len(pieces))
        bert_tokens.extend(pieces)
        curr_len_list.append(len(pieces))
        cum_num_list.append(cum_num)
        cum_num += len(pieces) - 1
    return bert_tokens, bert_label_action[
                        src_start:], src_start, orig_start, curr_len_list, cum_num_list


def _stage_1(bert_tokens, max_len):
    if len(bert_tokens) > max_len:
        new_len = max_len - (len(bert_tokens) - src_start)
        source_indices = list(range(new_len, max_len))
        bert_tokens = bert_tokens[:new_len] + bert_tokens[src_start:]
    else:
        new_len = src_start
        source_indices = list(range(src_start, len(bert_tokens)))
    return bert_tokens, new_len, source_indices


def _stage_2(orig_start, label_start, label_end, curr_len_list, cum_num_list, new_len):
    bert_label_start, bert_label_end = [], []
    bert_seq_width = []
    bert_rule = []
    cur_label_start, cur_label_end = [], []
    i = sum(seq_width[:orig_start])
    ptr = orig_start
    rem, sw, cur_len, cur_rule, ptr = upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
    while i < len(label_start):
        if rem > 0:
            st, ed = label_start[i], label_end[i]
            i += 1
            start = st + cum_num_list[st] if st < len(cum_num_list) else st
            end = ed + cum_num_list[ed] + curr_len_list[ed] - 1 if ed < len(cum_num_list) else ed
            if start >= new_len or end >= new_len:
                sw = max(1, sw - 1)
                start, end = 0, 0
            zeros = [0] * (cur_len - 1)
            cur_label_start.append([start] + zeros)
            cur_label_end.append([end] + zeros)
            rem -= 1
        if rem == 0:
            bert_seq_width.extend([sw] * cur_len)
            bert_rule.extend([cur_rule] * cur_len)
            for tup_s, tup_e in zip(zip(*cur_label_start), zip(*cur_label_end)):
                bert_label_start.append(tup_s)
                bert_label_end.append(tup_e)
            cur_label_start.clear()
            cur_label_end.clear()
            if ptr < len(curr_len_list):
                rem, sw, cur_len, cur_rule, ptr = upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
            else:
                assert (i == len(label_start))
    assert (len(bert_label_start) == len(bert_seq_width) == len(bert_rule))
    return bert_label_start, bert_label_end, bert_seq_width, bert_rule


if __name__ == '__main__':

    for item in dataset:
        print(item)
        break

    item = item['json']
    # context, src, tgt = item['context'], item['source'], item['target']

    # src = context + ' | ' + src
    src = '|'.join(item['source'])
    tgt = item['target']
    tgt = ' '.join(tgt.strip().split())
    tokens = [tokenizer.cls_token] + src.strip().split(' ')

    action_seq, start_seq, end_seq, rule_seq = tuple(item['actions']), item['starts'], item['ends'], item['rules']

    action_seq = [tag2idx.get(tag) for tag in ('DELETE',) + action_seq]
    start_seq, seq_width = _split_multi_span(start_seq)
    end_seq, _ = _split_multi_span(end_seq)
    rule_seq = [0] + [rules_vocab.index(item) for item in rule_seq]

    # rule_seq = [0] + list(map(to_int, rule_seq))

    # bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, bert_rule, src_indices = _split_to_wordpieces_span(
    #     tokens, action_seq, start_seq, end_seq, seq_width, rule_seq)
    bert_tokens, bert_label_action, src_start, orig_start, curr_len_list, cum_num_list = _split_to_wordpieces_span(
        tokens, action_seq)
    bert_tokens, new_len, source_indices = _stage_1(bert_tokens, max_len=512)
    _stage_2(orig_start, start_seq, end_seq, curr_len_list, cum_num_list, new_len)
