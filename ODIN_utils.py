from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, TypeVar, Optional, Callable, List
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torchvision.datasets import ImageFolder

from kaokore_ds import train_dataset
from pytorch_ood.model import WideResNet

MODEL_TYPE = ['stsaclf', 'simple_odin'][1]

class OdinSamplerRB(torch.utils.data.Sampler):
    def __init__(self, model, data_source, batch_size, step_sz, loss, temperature, norm_std, replacement=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.replacement = replacement

        self.step_sz = step_sz
        self.loss = loss
        self.temperature = temperature
        self.norm_std = norm_std

        self.model = model

        self.sampling_probs = np.zeros(len(data_source))

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

        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                if MODEL_TYPE == 'simple_odin':
                    logits = self.model(x) / temperature
                elif MODEL_TYPE == 'stsaclf':
                    outputs = self.model(x)
                    if isinstance(outputs, list):
                        logits = outputs[0] / temperature
                if y is None:
                    y = logits.max(dim=1).indices
                loss = criterion(logits, y)
                loss.backward()

                gradient = torch.sign(x.grad.data)

                if norm_std:
                    for i, std in enumerate(norm_std):
                        gradient.index_copy_(
                            1,
                            torch.LongTensor([i]).to(gradient.device),
                            gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
                        )

                x_hat = x - eps * gradient

        return x_hat

    def predict_confidence_probs(self, x: Tensor, x_s: Tensor, y: Tensor) -> Tensor:
        """
        Calculates softmax outlier scores on ODIN pre-processed inputs.

        :param x: input tensor
        :return: outlier scores for each sample
        """
        clses  = torch.unique(y)
        scores_class = {i:0 for i in torch.unique(y)}

        x_hat = self.odin_preprocessing(
            x=x,
            y=y,
            eps=self.step_sz,
            criterion=self.loss,
            temperature=self.temperature,
            norm_std=self.norm_std,
        )
        
        if MODEL_TYPE == 'simple_odin':
            results = self.model(x_hat).softmax(dim=1)
            aug_results = self.model(x_s).softmax(dim=1)
        elif MODEL_TYPE == 'stsaclf':
            results = self.model(x_hat)[0].softmax(dim=1)
            aug_results = self.model(x_s)[0].softmax(dim=1)
        
        #choosing to keep original or transformation - simple strat
        confidence_new = torch.zeros_like(results).float()
        results = torch.where(results>aug_results, results, aug_results)
        
        confidence = torch.tensor(results).max(dim=1).values
        confidence_probs = torch.zeros_like(y).float()
        for i in scores_class:
          scores_class[i] = confidence[torch.nonzero(y == i).squeeze()].sum()#make it positive
          confidence_probs[torch.nonzero(y == i).squeeze()] = scores_class[i]
        return confidence_probs, scores_class

    def __iter__(self):
        n = len(self.data_source)

        old_indices = torch.randperm(n).tolist()
        cum_scores_class = {i: 0 for i in [0,1,2,3]}
        train_egs = [2045, 1288, 432, 473] # to automate

        #train local model for ODIN, get aggregate score
        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
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
            
            probs, scores = self.predict_confidence_probs(x, x_s, y)
            for i in scores:
                cum_scores_class[i.item()]+= scores[i]
            print(cum_scores_class)
            break

        for i in cum_scores_class:
            cum_scores_class[i]/= train_egs[i]

        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
            x,y, x_s = [], [], []
            seed_idxs = old_indices[i:i+self.batch_size]
            
            for idx in seed_idxs:
              #x.append(F.interpolate(self.data_source[idx][0], size = 32))
              x.append(self.data_source[idx][0])
              x_s.append(self.data_source[idx][1])
              y.append(self.data_source[idx][2])

            if x == [] or y == []:
              print(seed_idxs, i)
            x = torch.stack(x).to('cuda')
            x_s = torch.stack(x_s).to('cuda')
            y  = torch.tensor(y).to('cuda')
            probs, _ = self.predict_confidence_probs(x, x_s, y)
            
            #rescaling
            cum_confidence = torch.zeros_like(y).float()
            for i in cum_scores_class:
                cum_confidence[torch.nonzero(y == i).squeeze()] = cum_scores_class[i]
            probs = probs * cum_confidence
            
            probs = np.array(probs.detach().cpu())
            probs = probs/np.sum(probs)#normalizing probs across batch

            #to choose low confidence samples
            self.sampling_probs[seed_idxs] = 1 - probs

            #stratified sampling
            indices = np.random.choice(seed_idxs, size=self.batch_size, replace=True, p=probs)

            #random sampling here, uncomment to test
            #indices = seed_idxs
            
            yield indices

    def vis_sampling_projections(self, data_loader, epoch):
      def get_embeddings(model, data_loader):
        embeddings = []
        model.eval()
        all_lbls = []
        with torch.no_grad():
            for inputs, lbls in data_loader:
                inputs = inputs.to(device)
                all_lbls.append(lbls)
                output = model(inputs)
                embeddings.append(output.cpu())
        return torch.cat(embeddings), torch.cat(all_lbls)

      # Get embeddings
      embeddings, labels = get_embeddings(self.model, data_loader)

      # Perform t-SNE dimensionality reduction
      tsne = TSNE(n_components=2)
      embeddings_tsne = tsne.fit_transform(embeddings)
      print(embeddings_tsne.shape, labels.shape)

      for class_label in np.unique(labels):
        indices = labels == class_label
        plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1],
                    alpha=self.sampling_probs[indices]*10+0.001, label=f'Class {class_label}')
        print(min(self.sampling_probs[indices]), max(self.sampling_probs[indices]))
        #plt.colorbar(label='Probabilities')
        plt.title('t-SNE Visualization with Class Color Coding and Probability Alpha')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
      plt.legend()
      plt.savefig(f'epoch_{epoch}.jpg')
      plt.show()
      plt.tight_layout()




    def __len__(self):
        return len(self.data_source) // self.batch_size


if __name__ == '__main__':
    #reproducibility
    # fix_random_seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.backends.cudnn.benchmark = False


    step_sz = 0.05
    criterion =  F.nll_loss
    device = 'cuda'
    batch_sz = 64

    epochs = 30
    lr = 1e-4
    momentum = 0.9
    decay = 0.0005

    temperature = 1

    dataset = train_dataset
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).to(device)
    art_model = model
    art_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True, device = 'cuda')
    trans = WideResNet.transform_for("cifar10-pt")
    norm_std = WideResNet.norm_std_for("cifar10-pt")

    batch_sampler = OdinSamplerRB(art_model, dataset, batch_sz,
                step_sz, criterion, temperature, norm_std)

    for i, batch_indices in enumerate(batch_sampler):
        break