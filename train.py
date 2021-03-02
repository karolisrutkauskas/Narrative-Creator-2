import timm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np

import dataset

# model_names = timm.list_models(pretrained=True)
# print(model_names)

model = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

transform = transforms.Compose(
    [
     transforms.ToPILImage(),
     transforms.Resize((384, 384)),
     transforms.ToTensor(),
     transforms.Lambda(dataset.make_img_rgb),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

trainset = dataset.NarrativesDataset(root='./data/images/', file='./data/dataset.jsonl', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        shuffle=True, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        print(data)
        inputs, labels = data
        # labels = torch.from_numpy(np.asarray(labels))

        inputs, labels = inputs.to(device), labels[0].to(device)

        # print(inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
print('Saving model...')

torch.save(model.state_dict(), './data/models/vit32-narr.pth')