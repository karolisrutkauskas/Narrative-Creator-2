import timm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np

from transformers import BartForConditionalGeneration as BCD, BartTokenizerFast as BTF

import dataset

import sys

batch_size = int(sys.argv[1])

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

trainset = dataset.NarrativesDataset(root='./data/images/', file='./data/dataset.jsonl', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

vit = vit.to(device)
bart = bart.to(device)
bart.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(vit.parameters()) + list(bart.parameters()), lr=3e-5)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # print(i)
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
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
print('Saving model...')

torch.save(vit.state_dict(), 'data/vit32-narr.pth')