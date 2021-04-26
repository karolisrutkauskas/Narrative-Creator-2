import timm
import torch
from transformers import BartForConditionalGeneration as BCD, BartTokenizerFast as BTF
from transformers.models.bart.modeling_bart import BartEncoder

vit = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=0)

image = torch.randn(2, 3, 384, 384)

o = vit.forward_features(image)

print(o.shape)

# o = vit(image)

# print(o.shape)

bart = BCD.from_pretrained('facebook/bart-base')
tokenizer = BTF.from_pretrained('facebook/bart-base')

tokenized_data = tokenizer.prepare_seq2seq_batch(["lol lol lol lol", "l", "l"], ["lole", "lole", "lole"], padding=True, truncation=True).data
# print(tokenized_data)

item = {key: torch.tensor(val[0]) for key, val in tokenized_data.items()}
item = torch.unsqueeze(item['input_ids'], 0)
print(item.shape)

# labels = torch.tensor(tokenized_data['labels'])
# print(labels.shape)

encoder = bart.get_encoder()

o = encoder(item)

print(o[0].shape)
