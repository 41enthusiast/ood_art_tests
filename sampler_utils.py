from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, TypeVar, Optional, Callable, List

import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets

from kaokore_ds import *
from stclf_model import *

#with and without style transfer
AUG_MODE = ['ST', 'VA'][1]
AGGR_MODE = ['Aggr','Indv','Both'][2]

class OdinSamplerRB(torch.utils.data.Sampler):
    def __init__(self, model, data_source, batch_size, step_sz, loss, temperature, norm_std, replacement=True, device = 'cuda'):
        self.data_source = data_source
        self.batch_size = batch_size
        self.replacement = replacement

        self.step_sz = step_sz
        self.loss = loss
        self.temperature = temperature
        self.norm_std = norm_std

        self.model = model

        self.sampling_probs = torch.zeros(len(data_source)).to(device)
        self.count_dict_new = []

        if len(data_source) < batch_size:
            raise ValueError("Batch size must be less than or equal to the dataset size.")

    def odin_preprocessing(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        criterion: Optional[Callable[[Tensor], Tensor]] = None,
        eps: float = 0.05,
        temperature: float = 1000,
        norm_std: Optional[List[float]] = None,
        mode = False
        ):
        """
        Functional version of ODIN.

        :param model: module to backpropagate through
        :param x: sample to preprocess
        :param y: the label :math:`\\hat{y}` which is used to evaluate the loss. If none is given, the models
            prediction will be used
        :param criterion: loss function :math:`\\mathcal{L}` to use. If none is given, we will use negative log
                likelihood
        :param eps: step size :math:`\\epsilon` of the gradient ascend step
        :param temperature: temperature :math:`T` to use for scaling
        :param norm_std: standard deviations used during preprocessing
        """

        # we make this assignment here, because adding the default to the constructor messes with sphinx
        if criterion is None:
            criterion = F.nll_loss
        model = self.model
        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                
                #temperature based softmax scaling for confidence calibration. doesn't affect model accuracy, but influences confidence.
                if MODEL_TYPE == 'simple_odin':
                    logits = model(x) / temperature
                elif MODEL_TYPE == 'stsaclf':
                    outputs = self.model(x)
                    if isinstance(outputs, list):
                        logits = outputs[0] / temperature
                if y is None:
                    y = logits.max(dim=1).indices
                loss = criterion(logits, y)
                loss.backward()

                if mode == True:
                    #gradient noise based data preprocessing
                    gradient = torch.sign(x.grad.data)

                    if norm_std:
                        for i, std in enumerate(norm_std):
                            gradient.index_copy_(
                                1,
                                torch.LongTensor([i]).to(gradient.device),
                                gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
                            )

                    x_hat = x - eps * gradient
        
        #this is in the training phase at the start of the epoch
        if mode == False:
            self.model = model
            
        if mode == True:
            return x_hat

    def predict_confidence_probs(self, x: Tensor, x_s: Tensor, y: Tensor, mode) -> Tensor:
        """
        Calculates softmax outlier scores on ODIN pre-processed inputs.

        :param x, x_s: input tensors
        :param y: output tensor
        :param mode: boolean to predict the confidence probabilities or not
        :return: outlier scores for each sample
        """

        x_hat = self.odin_preprocessing(
            x=x,
            y=y,
            eps=self.step_sz,
            criterion=self.loss,
            temperature=self.temperature,
            norm_std=self.norm_std,
            mode = mode
        )
        
        if mode == True:
            if MODEL_TYPE == 'simple_odin':
                results = self.model(x_hat).softmax(dim=1)
                aug_results = self.model(x_s).softmax(dim=1)
            elif MODEL_TYPE == 'stsaclf':
                results = self.model(x_hat)[0].softmax(dim=1)
                aug_results = self.model(x_s)[0].softmax(dim=1)
            
            #choosing to keep original or transformation - simple strat
            if AUG_MODE == 'ST':
                results = torch.where(results>aug_results, results, aug_results)
            
            confidence = torch.tensor(results).max(dim=1).values
            return confidence

    def update_local(self, model, temperature = 1):
        n = len(self.data_source)
        
        self.model = model
        self.temperature = temperature

        self.old_indices = torch.randperm(n).tolist()
        # print(self.old_indices)
        # print([self.data_source[i][2] for i in range(n)])
        
        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
            x,y, x_s = [], [], []
            seed_idxs = self.old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              x.append(self.data_source[idx][0])
              x_s.append(self.data_source[idx][1])
              y.append(self.data_source[idx][2])

            if x == [] or y == []:
              print(seed_idxs, i)
            x = torch.stack(x).to('cuda')
            x_s = torch.stack(x_s).to('cuda')
            y  = torch.tensor(y).to('cuda')
            self.predict_confidence_probs(x, x_s, y, False)
            
            
            

    def __iter__(self):
        n = len(self.data_source)

        old_indices = self.old_indices
        cum_scores_class = {i: 1e-6 for i in set(self.data_source.labels)}
        self.cum_sampling_probs = {i: 1e-6 for i in set(self.data_source.labels)}
        all_scores = []
        all_y = []

        #get aggregate score and individual scores. precalculate for the whole dataset
        print('Calculating confidence scores for the dataset')#the time consuming part of this code - finetuning
        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
            x,y, x_s = [], [], []
            seed_idxs = old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              x.append(self.data_source[idx][0])
              x_s.append(self.data_source[idx][1])
              y.append(self.data_source[idx][2])
              
            x = torch.stack(x).to('cuda')
            x_s = torch.stack(x_s).to('cuda')
            y  = torch.tensor(y).to('cuda')
            scores = self.predict_confidence_probs(x, x_s, y, True)
            all_y.append(y)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)
        self.all_y = torch.cat(all_y)
        print('Class wise statistics for analysis')
        for i in cum_scores_class:
            temp_group = all_scores[torch.nonzero(self.all_y == i).squeeze()] #big mistake here. wasn't an actual accumulation
            cum_scores_class[i]= temp_group.mean()
            
        print('Iterating through the dataset')
        self.count_dict_new.append({i:0 for i in set(self.data_source.labels)})
        for i in range(0, n, self.batch_size):
            x,y, x_s = [], [], []
            seed_idxs = old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              #x.append(F.interpolate(self.data_source[idx][0], size = 32))
              x.append(self.data_source[idx][0])
              x_s.append(self.data_source[idx][1])
              y.append(self.data_source[idx][2])
            
            x = torch.stack(x).to('cuda')
            x_s = torch.stack(x_s).to('cuda')
            y  = torch.tensor(y).to('cuda')
              
            probs = all_scores[seed_idxs]
            
            #rescaling
            cum_confidence = torch.zeros_like(y).float()
            for i in cum_scores_class:
                cum_confidence[torch.nonzero(y == i).squeeze()] = cum_scores_class[i]
            if AGGR_MODE == 'Both':
                probs = probs * cum_confidence # or cum confidence as mean, the indv probs as std dev instead of multiplyin it
            elif AGGR_MODE == 'Aggr':
                probs = cum_confidence
            #need to put higher confidence on the low confidence samples - it doesn't affect it in the end
            probs = 1 - probs

            #to choose low confidence samples
            self.sampling_probs[seed_idxs] =probs
            for i in cum_scores_class:
                self.cum_sampling_probs[i] += probs[torch.nonzero(y == i).squeeze().detach().cpu()].sum()

            #weighted sampling. Changed to not require the probabilities to be normalized
            #indices for the probs tensor - need to remap to seed indices
            indices = torch.multinomial(probs, num_samples=self.batch_size, replacement=True)
            for i in y[torch.unique(indices)]:
                self.count_dict_new[-1][i.item()]+=1

            og_indices = torch.tensor(seed_idxs, device= device)
            #print(indices, og_indices[indices], len(indices), self.batch_size)
            
            yield og_indices[indices] #major change here, mapped indices properly now

    def __len__(self):
        return len(self.data_source) // self.batch_size



def get_class_distribution(dataloader_obj, dataset_obj, split):
    
    count_dict = {i: 0 for i in set(dataset_obj.labels)}
    
    if split == 'train':
        for _, __, lbl in dataloader_obj:
            for l in lbl:
                count_dict[l.item()] += 1
    else:
        for _, lbl in dataloader_obj:
            for l in lbl:
                count_dict[l.item()] += 1
            
    return count_dict


np.random.seed(0)
torch.manual_seed(0)

dataset_size = len(train_dataset)
dataset_indices = list(range(dataset_size))
#uniform random sampling with replacement
print('Making a random sampler')
dataset_indices = np.random.choice(dataset_indices, size = dataset_size)
train_sampler = SubsetRandomSampler(dataset_indices)

#weighted sampler
print('Making a weighted sampler')
class_count = [i for i in train_dataset.count_dict.values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights)# simple form of aggregate
weighted_train_sampler = WeightedRandomSampler(weights = class_weights[train_dataset.labels],
                                               num_samples = len(train_dataset),
                                               replacement = True)

#random shuffling
BSZ = 64
train_loader_rs = DataLoader(dataset = train_dataset, shuffle = False, batch_size = BSZ,
                             sampler = train_sampler)
train_loader_ws = DataLoader(dataset = train_dataset, shuffle = False, batch_size = BSZ,
                             sampler = weighted_train_sampler)
print(get_class_distribution(train_loader_rs, train_dataset, 'train'))
print(get_class_distribution(train_loader_ws, train_dataset, 'train'))


print('Making the ODIN sampler')
temperature = 1
dataset = train_dataset
batch_sz = BSZ
device = 'cuda'
step_sz = 0.05
art_model = stclf_model.to(device)
MODEL_TYPE = ['stsaclf', 'simple_odin'][0]
batch_sampler = OdinSamplerRB(art_model, dataset, batch_sz,
              step_sz, F.nll_loss, temperature, norm_std)
batch_sampler.update_local(art_model, temperature)
# for i, batch_indices in enumerate(batch_sampler):
#     batch_x, batch_xs, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][2] for idx in batch_indices])
#     x,y = batch_x.cuda(), batch_y.cuda()
# print(batch_sampler.sampling_probs.mean(), batch_sampler.sampling_probs.std())
# print(batch_sampler.count_dict_new)

