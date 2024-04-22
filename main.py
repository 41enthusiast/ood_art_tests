import time
import csv
import os
import math
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch import Tensor

from datasets import load_dataset

from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, is_unknown
from pytorch_ood.detector import ODIN, odin_preprocessing

fix_random_seed(42)


from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from typing import Dict, TypeVar, Optional, Callable, List
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

import matplotlib.pyplot as plt
from pytorch_ood.utils import ToUnknown



from kaokore_ds import *
from utils import *
from ODIN_utils import *
from stclf_model import *

#just to clean up, but ill advised
import warnings
warnings.filterwarnings("ignore")

#presets
device = 'cuda'
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).to(device)

MODEL_TYPE = ['simple_odin', 'stsaclf', 'contrastive'][0]

if MODEL_TYPE == 'simple_odin':
  art_model = model
  art_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True, device = 'cuda')
elif MODEL_TYPE == 'stsaclf':
  art_model = stclf_model.to(device)



step_sz = 0.05
criterion = torch.nn.CrossEntropyLoss()
device = 'cuda'
batch_sz = 16

epochs = 30
lr = 1e-4
momentum = 0.9
decay = 0.0005

temperature = 1

dataset = train_dataset

batch_sampler = OdinSamplerRB(art_model, dataset, batch_sz,
              step_sz, F.nll_loss, temperature, norm_std)

optimizer = torch.optim.SGD(
   art_model.parameters(),
   lr, momentum=momentum,
   weight_decay=decay, nesterov=True)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        epochs * len(train_dataset)//batch_sz,
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / lr))

if MODEL_TYPE == 'stsaclf':
  lr = LR
  decay = WD
  epochs = EPOCHS
  criterion = focal_loss(4, 2, 2)
  optimizer = optim.Adam(art_model.parameters(), lr=lr)

#TRAINING LOOP

loss_avg = 0.0
temps = []
for epoch in range(epochs):
  art_model.train()
  for i, batch_indices in enumerate(batch_sampler):
    batch_x, batch_xs, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][2] for idx in batch_indices])
    x,y = batch_x.cuda(), batch_y.cuda()
  # for x, xs, y in train_loader_out:
  #   x, xs, y = x.to(device), y.to(device)
    if MODEL_TYPE == 'simple_odin':
      out = art_model(x)
    elif MODEL_TYPE == 'stsaclf':
      outputs = art_model(x)
      if isinstance(outputs, list):
          out = outputs[0]
    optimizer.zero_grad()
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_avg = loss_avg*0.8 + float(loss)* 0.2


  # Evaluation
  art_model.eval()
  all_labels = []
  all_preds = []
  with torch.no_grad():
      for inputs, labels in test_loader_out:
          if MODEL_TYPE == 'simple_odin':
            outputs = art_model(inputs.cuda())
            _, preds = torch.max(outputs, 1)
          elif MODEL_TYPE == 'stsaclf':
            outputs = art_model(inputs.cuda())
            if isinstance(outputs, list):
                preds = outputs[0].argmax(dim=1)
          
          all_labels.extend(labels.numpy())
          all_preds.extend(preds.cpu().numpy())

  # Calculate metrics
  accuracy = accuracy_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds, average='weighted')
  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')

  print(f'Epoch [{epoch+1}/{epochs}], '
        f'Accuracy: {accuracy:.4f}, '
        f'F1 Score: {f1:.4f}, '
        f'Precision: {precision:.4f}, '
        f'Recall: {recall:.4f}')

  #batch_sampler.vis_sampling_projections(train_loader_mini, epoch)
  batch_sampler.model = art_model
  if epoch %5 == 0:
    batch_sampler.temperature *= 5
    # batch_sampler.temperature = cosine_annealing(epoch, len(train_dataset)//batch_sz, 0, 1000)
    # temps.append(batch_sampler.temperature)
    #print('temperature tests: ', batch_sampler.temperature)

print('train_loss', loss_avg)
print(temps)
