import timm
from timm.utils import CheckpointSaver
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np

from transformers import BartForConditionalGeneration as BCD, BartTokenizerFast as BTF

import dataset

import sys

def run_train(batch_size, learn_rate, number_of_epochs):
    vit = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    bart = BCD.from_pretrained('facebook/bart-base')
    tokenizer = BTF.from_pretrained('facebook/bart-base')

    transform = transforms.Compose(
        [
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Lambda(dataset.make_img_rgb),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    data_set = dataset.NarrativesDataset(root='./data/images/', file='./data/dataset.jsonl', transform=transform)

    train_set_fraction = int(len(data_set) * 0.8)

    train_set, val_set = torch.utils.data.random_split(dataset=data_set, lengths=[train_set_fraction, len(data_set) - train_set_fraction])

    print(len(train_set))
    print(len(val_set))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    vit = vit.to(device)
    vit.train()
    bart = bart.to(device)
    bart.train()
    
    optimizer = optim.Adam(list(vit.parameters()) + list(bart.parameters()), lr=learn_rate)

    cs = CheckpointSaver(vit, optimizer, checkpoint_dir='data/checkpoints')

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        number_of_iterations = 0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            images, narratives = data
            images = images.to(device)

            image_features = vit.forward_features(images)
            tokenized_data = tokenizer.prepare_seq2seq_batch("", list(narratives), padding=True, truncation=True).data
            labels = torch.tensor(tokenized_data['labels']).to(device)

            image_features = torch.unsqueeze(torch.unsqueeze(image_features, 1), 0).to(device)

            bart_outputs = bart(encoder_outputs=image_features, labels=labels)
            
            loss = bart_outputs[0]
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            number_of_iterations += 1
        print('Epoch {} finished, loss: {}'.format(epoch, running_loss / number_of_iterations))
        print('Running eval...')
        run_eval(vit, bart, valloader, device, tokenizer)

    print('Finished Training')
    print('Saving model...')

    cs.save_checkpoint(1)
    bart.save_pretrained('data/bart')

def run_eval(vit, bart, eval_loader, device, tokenizer):
    vit.eval()
    bart.eval()
    
    running_loss = 0.0
    number_of_iterations = 0

    with torch.no_grad():
        for i, data in enumerate(eval_loader, 0):
            images, narratives = data
            images = images.to(device)

            image_features = vit.forward_features(images)
            tokenized_data = tokenizer.prepare_seq2seq_batch("", list(narratives), padding=True, truncation=True).data
            labels = torch.tensor(tokenized_data['labels']).to(device)

            image_features = torch.unsqueeze(torch.unsqueeze(image_features, 1), 0).to(device)

            bart_outputs = bart(encoder_outputs=image_features, labels=labels)
            
            loss = bart_outputs[0]

            running_loss += loss.item()
            number_of_iterations += 1
    
    print('Eval finished, loss: {}'.format(running_loss / number_of_iterations))

    vit.train()
    bart.train()

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    learn_rate = float(sys.argv[2])
    number_of_epochs = int(sys.argv[3])
    run_train(batch_size, learn_rate, number_of_epochs)