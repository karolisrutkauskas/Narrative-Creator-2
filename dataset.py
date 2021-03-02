from torch.utils.data import Dataset

from skimage import io
import json

class NarrativesDataset(Dataset):
    def __init__(self, root, file, transform=None):
        self.root = root
        self.data = read_data(file)
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        img_id, narr = self.data[id]
        image = io.imread(self.root + img_id + '.jpg')
        if self.transform is not None:
            image = self.transform(image)
            # print(image.shape)
            # print(image)
            # image = image.convert("RGB")'

        # narr = torch.Tensor(narr).int()

        return image, narr


def read_data(file):
    data_pairs = []
    with open(file, 'r') as jsonl_file:
        json_list = list(jsonl_file)
        for item in json_list:
            result = json.loads(item)
            image_id = result['image_id']
            narrative = result['narrative']
            data_pairs.append((image_id, narrative))

    return data_pairs

def make_img_rgb(x):
    if list(x.shape)[0] == 1:
        return x.repeat(3, 1, 1)
    else: return x



###########################################
import torch
import torchvision.transforms as transforms

if __name__ == "__main__":
    transform = transforms.Compose(
    [
     transforms.ToPILImage(),
     transforms.Resize((384, 550)),
     transforms.ToTensor(),
     transforms.Lambda(make_img_rgb),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

    trainset = NarrativesDataset(root='./data/images/', file='./data/dataset.jsonl', transform=transform)
                                            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

    for i, data in enumerate(trainloader, 0):
        img, narr = data
        print(img)
#############################################