import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import imageio

from transformers import BartForConditionalGeneration as BCD, BartTokenizerFast as BTF, ViTModel, ViTFeatureExtractor

import dataset

import sys

def run_train(batch_size, learn_rate, number_of_epochs, do_eval):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    vit = ViTModel.from_pretrained('google/vit-base-patch16-384')
    # freeze_vit(vit)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    bart = BCD.from_pretrained('facebook/bart-base')
    freeze_bart(bart)
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

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        number_of_iterations = 0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            images, narratives = data
            images = to_list(images)

            for i, image in enumerate(images):
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=2)
                    image = np.repeat(image, 3, 2)
                    images[i] = image

            inputs = feature_extractor(images=images, return_tensors="pt").to(device)
            image_features = vit(**inputs).last_hidden_state

            tokenized_data = tokenizer.prepare_seq2seq_batch("", list(narratives), padding=True, truncation=True).data
            labels = torch.tensor(tokenized_data['labels']).to(device)

            image_features = torch.unsqueeze(image_features, 0).to(device)

            bart_outputs = bart(encoder_outputs=image_features, labels=labels)
            
            loss = bart_outputs[0]
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            number_of_iterations += 1
        print('Epoch {} finished, loss: {}'.format(epoch, running_loss / number_of_iterations))

        if do_eval:
            print('Running eval...')
            run_eval(vit, feature_extractor, bart, valloader, device, tokenizer)
            
    print('Finished Training')
    print('Saving model...')

    vit.save_pretrained('data/vit')
    bart.save_pretrained('data/bart')

def run_eval(vit, feature_extractor, bart, eval_loader, device, tokenizer):
    vit.eval()
    bart.eval()
    
    running_loss = 0.0
    number_of_iterations = 0

    with torch.no_grad():
        for i, data in enumerate(eval_loader, 0):
            images, narratives = data
            images = to_list(images)

            inputs = feature_extractor(images=images, return_tensors="pt").to(device)
            image_features = vit(**inputs).last_hidden_state
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

def freeze_vit(vit):
    for child in vit.named_children():
        if child[0] == 'blocks':
            number_of_blocks = int(len(child[1]) / 2)

            for i in range(number_of_blocks):
                for param in child[1][i].parameters():
                    param.requires_grad = False

def freeze_bart(bart):
    for child in bart.get_decoder().named_children():
        if child[0] == 'layers':
            number_of_blocks = int(len(child[1]) / 2)

            for i in range(number_of_blocks, len(child[1])):
                for param in child[1][i].parameters():
                    param.requires_grad = False

def to_list(array):
    image_list = []
    for image_id in array:
        image = imageio.imread('data/images/' + image_id + '.jpg')
        image_list.insert(0, image)

    return image_list

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    learn_rate = float(sys.argv[2])
    number_of_epochs = int(sys.argv[3])
    do_eval = bool(sys.argv[4])
    run_train(batch_size, learn_rate, number_of_epochs, do_eval)