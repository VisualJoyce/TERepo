import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoModelForCausalLM, \
    BartForCausalLM, BartModel
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
# model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-base-chinese")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
# text = "北京是的首都"
text = "这的确是个好办[MASK]"
print(tokenizer(text))
print(text2text_generator(text, max_length=len(text)*2, do_sample=True))

tokenizer = BertTokenizer.from_pretrained(os.path.expanduser("~/Data/pretrained_models/text_editing/gec-zh-bart-large-chinese"))
# model = AutoModelForCausalLM.from_pretrained("/home/tanminghuan/Data/pretrained_models/text_editing/gec-zh-bart-large-chinese")
model = BartForConditionalGeneration.from_pretrained(os.path.expanduser("~/Data/pretrained_models/text_editing/gec-zh-bart-large-chinese"))
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(tokenizer(text))
print(text2text_generator(text, max_length=len(text)*2, do_sample=False))

# outputs = model.generate(max_length=40)  # do greedy decoding
# print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

input_ids = tokenizer.encode(text, return_tensors='pt')  # encode input context
outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
for i in range(3): #  3 output sequences were generated
    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))


# model = BartModel.from_pretrained(os.path.expanduser("~/Data/pretrained_models/text_editing/gec-zh-bart-large-chinese"))
