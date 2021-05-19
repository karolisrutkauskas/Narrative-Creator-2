import json
from typing import Any, Dict
from types import MethodType
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
from skimage import io
import torchvision.transforms as transforms
import dataset
import timm
import sys
from tqdm import tqdm

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


def run_generation(dataset_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BartForConditionalGeneration.from_pretrained('data/bart').to(device)

    model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, model)
    model._prepare_encoder_decoder_kwargs_for_generation = MethodType(_prepare_encoder_decoder_kwargs_for_generation, model)

    vit = timm.create_model('vit_base_patch32_384').to(device)
    timm.models.helpers.load_checkpoint(vit, 'data/checkpoints/last.pth.tar', strict=False)

    transform = transforms.Compose(
        [
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Lambda(dataset.make_img_rgb),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    test_set = dataset.NarrativesDataset(root='./data/test_img/', file='./data/dataset_test.jsonl', transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                            shuffle=True, num_workers=2)

    expected_narratives = []
    generated_narratives = []

    
    for i, data in tqdm(enumerate(testloader, 0)):
        image, expected_narrative = data

        image = image.to(device)

        image_features = vit.forward_features(image)
        image_features = torch.unsqueeze(torch.unsqueeze(image_features, 1), 0)

        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

        batch = tokenizer.prepare_seq2seq_batch([""], return_tensors="pt")

        image_features = image_features.repeat(1, 4, 1, 1)

        bart_outputs = model.generate(**batch, encoder_output=image_features, max_length=128)

        narrative = tokenizer.batch_decode(bart_outputs, skip_special_tokens=True)

        expected_narratives.insert(0, expected_narrative[0])
        generated_narratives.insert(0, narrative[0])

    with open('data/narr_pairs.jsonl', 'a') as jsonl_file:
        for i, narrative in enumerate(expected_narratives):
            data = {'expected': expected_narratives[i], 'generated': generated_narratives[i]}
            jsonl_file.write(json.dumps(data))
            jsonl_file.write('\n')


if __name__ == '__main__':
    dataset_file = sys.argv[1]
    run_generation(dataset_file)