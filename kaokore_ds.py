from sklearn.model_selection import train_test_split
import csv
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_ood.model import WideResNet
import torch
from torchvision import datasets, models, transforms
import pandas as pd

#from rois-codh code folder code for pytorch
def verify_str_arg(value, valid_values):
    assert value in valid_values
    return value


def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
                for row in reader]

def stratify_split_ds(dataset, split_pct = 0.1, shuffle = True):
  ds_idx = train_test_split(np.arange(len(dataset)),
                                             train_size = split_pct,
                                             random_state=42,
                                             shuffle=shuffle,
                                             stratify=dataset.labels)[0]
  return ds_idx

def classwise_split_ds(dataset, split_pct = 0.1, shuffle = True):
  ds_idx = train_test_split(np.arange(len(dataset)),
                                             train_size = split_pct,
                                             random_state=42,
                                             shuffle=shuffle,
                                             stratify=dataset.labels)[0]
  return ds_idx

class Kaokore(Dataset):

    def __init__(self, root, split='train', category='gender', transform=None, label_type = 'known'):
        self.root = root = os.path.expanduser(root)

        self.split = verify_str_arg(split, ['train', 'dev', 'test'])

        self.category = verify_str_arg(category, ['gender', 'status'])

        labels = load_labels(os.path.join(root, 'labels.csv'))

        self.num_classes, self.txt_lbls = self.get_metadata(category)
        self.count_dict = {i:0 for i in range(self.num_classes)}
        
        self.entries = []; self.labels = []
        
        for label_entry in labels:
          if label_entry['set'] == split and os.path.exists(
              os.path.join(self.root, 'images_256', label_entry['image'])):
            self.entries.append((label_entry['image'], int(label_entry[category])))
            self.labels.append(int(label_entry[category]))
            self.count_dict[self.labels[-1]]+=1
            

        self.transform = transform

    def get_metadata(self, category):
      f = open(os.path.join(self.root, 'labels.metadata.en.txt'), 'r')
      fout = f.read().split('\n\n')
      keys, vals = fout[0].split('\n')
      num_classes_map = lambda k,v : (k.split('\t'), v.split('\t'))
      num_classes = {ki:int(vi) for ki,vi in zip(*num_classes_map(keys, vals))}[category]
      for txtlbls in fout[1:]:
        if category in txtlbls:
          txtlbls = txtlbls.split('\n\t')[1:]
          txtlbls = {int(i.split('\t')[0]) : i.split('\t')[1] for i in txtlbls}
          break
      return num_classes, txtlbls
          
    
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image_filename, label = self.entries[index]

        image_filepath = os.path.join(self.root, 'images_256', image_filename)
        image = image_loader(image_filepath)
        if self.split == 'train':
          stylized_filepath = os.path.join('class_styled_kaokore/allinone', image_filename)
          stylized = image_loader(stylized_filepath)
        if self.transform is not None:
            image = self.transform(image)
            if self.split == 'train':
              stylized = self.transform(stylized)
        if self.split == 'train':
          return image, stylized, label
        else:
          return image, label
    

# def make_stratified_

def make_kaokore_df(dset: Kaokore):
  kao_df = []
  for i in dset.entries:
    kao_df.append({'image':os.path.join(dset.root, 'images_256', i[0]),
                   'label':dset.txt_lbls[i[1]]
                   })
  return pd.DataFrame(kao_df)
  

#hyperparameters setup
split_pct = 0.5
label_type = 'status'
BSZ = 32
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")
if label_type == 'status':
  num_classes = 4
else:
  num_classes = 2

# prepare dataset with pretraining dataset statistics
train_dataset = Kaokore(os.path.join('kaokore','kaokore'), 'train', label_type, trans, 'known')
train_idx = stratify_split_ds(train_dataset, split_pct)
train_dl = DataLoader(train_dataset, batch_size = BSZ)
test_dataset = Kaokore(os.path.join('kaokore','kaokore'), 'test', label_type, trans, 'known')
test_idx = stratify_split_ds(test_dataset, split_pct)
test_dl = DataLoader(test_dataset, batch_size = BSZ)

#calc mean and std deviation of kaokore images
num_images = len(train_dataset)
sum_channels = torch.zeros(3)
sum_squares_channels = torch.zeros(3)
for image, style, y in train_dl:
    sum_channels += torch.sum(image, dim=(0, 2, 3))  # Sum across height and width dimensions
    sum_squares_channels += torch.sum(image ** 2, dim=(0, 2, 3))  # Sum of squares across height and width dimensions
mean_channels = sum_channels / (num_images * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2])
std_channels = torch.sqrt((sum_squares_channels / (num_images * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2])) - mean_channels ** 2)

kaokore_transform1 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(tuple(mean_channels.tolist()), tuple(std_channels.tolist())),
])
kaokore_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(tuple(mean_channels.tolist()), tuple(std_channels.tolist())),
])

train_dataset = Kaokore(os.path.join('kaokore','kaokore'), 'train', label_type, kaokore_transform1, 'known')
train_dataset_mini = Kaokore(os.path.join('kaokore','kaokore'), 'train', label_type, kaokore_transform, 'known')
test_dataset = Kaokore(os.path.join('kaokore','kaokore'), 'test', label_type, kaokore_transform1, 'known')

train_loader_out = DataLoader(train_dataset, batch_size = BSZ)
train_loader_mini = DataLoader(train_dataset_mini, batch_size = BSZ, shuffle = False)
test_loader_out = DataLoader(test_dataset, batch_size = BSZ)

if __name__ == '__main__':
  print(train_dataset.count_dict, test_dataset.count_dict)
  for x, xs, y in train_loader_out:
    break
  for x, y in test_loader_out:
    break