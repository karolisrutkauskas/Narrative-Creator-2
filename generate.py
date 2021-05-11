from typing import Any, Dict
from types import MethodType
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
from skimage import io
import torchvision.transforms as transforms
import dataset
import timm

def prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past=None,
    attention_mask=None,
    head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs):
    # cut decoder_input_ids if past is used
    if past is not None:
        decoder_input_ids = decoder_input_ids[:, -1:]

    inputs = {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
        "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    }

    inputs.pop('attention_mask')
    inputs["encoder_outputs"] = kwargs["encoder_output"]

    return inputs

def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
    # retrieve encoder hidden states
    encoder = self.get_encoder()
    encoder_kwargs = {
        argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and argument != "encoder_output"
    }
    print(encoder_kwargs.keys())
    model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
    return model_kwargs

model = BartForConditionalGeneration.from_pretrained('data/bart')

model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, model)
model._prepare_encoder_decoder_kwargs_for_generation = MethodType(_prepare_encoder_decoder_kwargs_for_generation, model)

# model = bart_decorator.BartDecorator(model)


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

image = io.imread('data/images/fc06c1b59dd8ca6e.jpg')
image = transform(image)
image = torch.unsqueeze(image, 0)

image_features = vit.forward_features(image)
image_features = torch.unsqueeze(torch.unsqueeze(image_features, 1), 0)

print(image_features.shape)

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

# batch = tokenizer.prepare_seq2seq_batch([""], tgt_texts=["In this image I can see on the left side it is a house. In the middle it looks like a tower, on the right side there are trees, at the top it is the sky."], return_tensors="pt")
batch = tokenizer.prepare_seq2seq_batch([""], return_tensors="pt")

print(batch)

# print(torch.unsqueeze(torch.tensor(batch['input_ids']), 0).shape)

# bart_outputs = model(batch.input_ids, encoder_outputs=image_features)
# print(bart_outputs.keys())
bart_outputs = model.generate(**batch, encoder_output=image_features, num_beams=1, max_length=128)

narrative = tokenizer.batch_decode(bart_outputs, skip_special_tokens=True)

print(narrative)
