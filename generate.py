from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
from skimage import io
import torchvision.transforms as transforms
import dataset
import timm

model = BartForConditionalGeneration.from_pretrained('data/bart')
vit = timm.create_model('vit_base_patch32_384')
timm.models.helpers.load_checkpoint(vit, 'data/checkpoints/last.pth.tar', strict=False)

transform = transforms.Compose(
    [
     transforms.ToPILImage(),
     transforms.Resize((384, 384)),
     transforms.ToTensor(),
     transforms.Lambda(dataset.make_img_rgb),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

image = io.imread('data/images/4839f5eac98771bf.jpg')
image = transform(image)
image = torch.unsqueeze(image, 0)

image_features = vit.forward_features(image)
image_features = torch.unsqueeze(torch.unsqueeze(image_features, 1), 0)

print(image_features.shape)

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

batch = tokenizer.prepare_seq2seq_batch("").data

bart_outputs = model(torch.unsqueeze(torch.tensor(batch['input_ids']), 0), encoder_outputs=image_features)

narrative = tokenizer.batch_decode(bart_outputs[1], skip_special_tokens=True)

print(narrative)
