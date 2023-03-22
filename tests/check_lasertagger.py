from glob import glob

import webdataset as wds

from terepo.models.tagging import LaserTaggerTokenizer, GECToRConfig, GECToRTokenizer

config_0 = GECToRConfig.from_pretrained(
    "/home/vimos/Data/Text/pretrained_models/text_editing/gec-en-gector-bert-base-cased")

tokenizer_0 = GECToRTokenizer.from_pretrained(
    "/home/vimos/Data/Text/pretrained_models/text_editing/gec-en-gector-bert-base-cased")

tokenizer = LaserTaggerTokenizer.from_pretrained(
    "/home/vimos/Data/Text/pretrained_models/text_editing/wikisplit-en-lasertager-bert-base-cased")
dataset = wds.WebDataset(
    glob("/home/vimos/git/WenXin/data/annotations/text_editing/en/wikisplit/train/*.tar")).decode()

from collections import Counter


def parse(source_sent, target_sent):
    source_tokens = tokenizer_0.convert_sequence_to_tokens(source_sent)
    target_tokens = tokenizer_0.convert_sequence_to_tokens(target_sent)
    edits = list(tokenizer_0.editor.convert_sequences_to_edits(source_tokens, target_tokens))
    labels = tokenizer_0.convert_edits_into_labels_list(source_tokens, edits)
    print('--------------------------------------')
    print(source_sent)
    print(len(source_tokens), source_tokens)
    print(target_sent)
    print(len(target_tokens), target_tokens)

    print(edits)
    print(len(labels), labels)


c = Counter()
for i, item in enumerate(dataset):
    source = item['json']['source']
    target = item['json']['target']
    if source == target:
        continue
    print(item)
    source_tokens = tokenizer.convert_sequence_to_tokens(source)
    target_tokens = tokenizer.convert_sequence_to_tokens(target)
    print(source_tokens)
    print(target_tokens)

    parse(source, target)

    # edits = list(tokenizer.editor.convert_sequences_to_edits(source_tokens, target_tokens))
    # labels_list = tokenizer.convert_edits_into_labels_list(source_tokens, edits)
    # label_ids, detect_tags = tokenizer.convert_labels_list_to_ids(labels_list)
    # c.update([len(n) for n in labels_list])
    # print(label_ids)
    tags = tokenizer.editor._compute_tags_fixed_order(source_tokens, target_tokens)
    print([str(tag) for tag in tags])
    if i > 50:
        break
