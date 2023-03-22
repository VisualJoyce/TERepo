from transformers import AutoTokenizer

from terepo.models.tagging import GECToRConfig
from terepo.models.tagging.gector.feature_extraction_gector import GECToRFeatureExtractor

config = GECToRConfig.from_pretrained("aihijo/gec-en-gector-roberta-base")

print(config)

tokenizer = AutoTokenizer.from_pretrained("aihijo/gec-en-gector-roberta-base", config=config.encoder)

feature_extractor = GECToRFeatureExtractor.from_pretrained("aihijo/gec-en-gector-roberta-base")

# tokenizer = GECToRTokenizer.from_pretrained(
#     "/home/tanminghuan/Data/pretrained_models/text_editing/gec-en-gector-bert-base-cased-v2/")


def parse(source_sent, target_sent):
    source_tokens = feature_extractor.convert_sequence_to_tokens(source_sent, tokenizer)
    target_tokens = feature_extractor.convert_sequence_to_tokens(target_sent, tokenizer)
    edits = list(feature_extractor.editor.convert_sequences_to_edits(source_tokens, target_tokens))
    labels = feature_extractor.convert_edits_into_labels_list(source_tokens, edits)
    print('--------------------------------------')
    print(source_sent)
    print(len(source_tokens), source_tokens)
    print(target_sent)
    print(len(target_tokens), target_tokens)

    print(edits)
    print(len(labels), labels)

    label_ids, _ = feature_extractor.convert_labels_list_to_ids(labels)
    print(label_ids)

    target_line = feature_extractor.convert_labels_list_to_sentence(source_tokens, labels)
    print(len(target_line), target_line)


# source_sent = "We trying scores more but we worked together of 90 minutes ."
# # target_sent = "We tried to score more but we worked together for 90 minutes ."
# target_sent = "We trying to score more but we worked together for 90 minutes ."
# parse(source_sent, target_sent)
# gold_edits = convert_para_to_edits([source_sent], [target_sent])
# print(source_sent)
#
# with open('../GEC/synthetic/a1/a1_train_incorr_sentences.txt') as sf:
#     with open('../GEC/synthetic/a1/a1_train_corr_sentences.txt') as tf:
#         for source_sent, target_sent in zip(sf, tf):
#             flag = 0
#             parse(source_sent, target_sent)

item = {'__key__': 'wi+locness_ABCN_dev_gold_bea19-23', '__url__': 'wi+locness/dev/00000000.tar', 'json': {
    'source': 'I love children , and I enjoy looking after them . also , I organized many sports activities before in my school .',
    'target': 'I love children , and I enjoy looking after them . Also , I organized many sports activities before at my school .'}}

item = {
    "json": {
        "source": "Nowadays there are many people that are learning foreign language , for me is a good thing that more people learn this , but is it worth learning a foreign language ?",
        "target": "Nowadays , there are many people who are learning foreign languages . For me , it is a good thing that more people are learning languages , but is it worth learning a foreign language ?"
    }
}

parse(item['json']['source'], item['json']['target'])
