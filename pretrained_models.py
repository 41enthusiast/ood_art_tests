import torch
from torch import nn
from torchvision import models
from collections import namedtuple
from collections import OrderedDict
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from typing import Type, Any, Union, List

from torch import nn
import torch.nn.functional as F

from collections import namedtuple


from torchvision import models
from torchvision.models import VGG16_Weights


def get_project_in(model: torch.nn.Sequential):
    out_dim = 0
    for layer in model.children():
        if 'Conv2d' in str(type(layer)):
            out_dim = layer.out_channels
        elif 'BasicBlock' in str(type(layer)):
            out_dim = get_project_in(layer)
        elif 'Bottleneck' in str(type(layer)):
            out_dim = get_project_in(layer)
        elif 'Sequential' in str(type(layer)):
            out_dim = get_project_in(layer)
    return out_dim

#ResNet-N
class IntResNet(ResNet):
    def __init__(self,output_layer, output_layers, *args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        self.project_in = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

        for lyr in self.layers:
          if lyr in output_layers:
            if 'Conv2d' in str(type(self.layers[lyr])):
              self.project_in.append(self.layers[lyr].out_channels)   
            elif 'Sequential' in str(type(self.layers[lyr])):
              self.project_in.append(get_project_in(self.layers[lyr]))

        self.output_layers = output_layers

    def _forward_impl(self, x):
        outputs = []
        for l in self._layers:
            x = self.layers[l](x)
            if l in self.output_layers:
              outputs.append(x)

        return outputs, x

    def forward(self, x):
        return self._forward_impl(x)

class ResNetN(nn.Module):
    # base_model : The model we want to get the output from
    # base_out_layer : The layer we want to get output from
    # num_trainable_layer : Number of layers we want to finetune (counted from the top)
    #                       if enetered value is -1, then all the layers are fine-tuned
    def __init__(self,base_model,base_out_layer, base_out_lyrs, requires_grad = False):
        super().__init__()
        self.base_model = base_model
        self.base_out_layer = base_out_layer
        self.base_out_lyrs = base_out_lyrs
        
        self.model_dict = {
                           'resnet34':{'block':BasicBlock,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet50':{'block':Bottleneck,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet101':{'block':Bottleneck,'layers':[3,4,23,3],'kwargs':{}},
                           'resnet152':{'block':Bottleneck,'layers':[3,8,36,3],'kwargs':{}},
                           }
        
        #PRETRAINED MODEL
        self.resnet = self.new_resnet(self.base_model,self.base_out_layer,self.base_out_lyrs,
                                     self.model_dict[self.base_model]['block'],
                                     self.model_dict[self.base_model]['layers'],
                                     True,True,
                                     **self.model_dict[self.base_model]['kwargs'])

        self.layers = list(self.resnet._modules.keys())#has the truncated model
        print(self.layers)
        self.project_ins = self.resnet.project_in
        #FREEZING LAYERS
        self.total_children = 0
        self.children_counter = 0
        for c in self.resnet.children():
            self.total_children += 1
        
        for c in self.resnet.children():
            for param in c.parameters():
                param.requires_grad = requires_grad
            self.children_counter += 1
                    
    def new_resnet(self,
                   model_type: str,
                   outlayer: str,
                   outLayers: List[str],
                   block: Type[Union[BasicBlock, Bottleneck]],
                   layers: List[int],
                   pretrained: bool,
                   progress: bool,
                   **kwargs: Any
                  ) -> IntResNet:

        '''model_urls = {
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth'
        }'''
        model = IntResNet(outlayer, outLayers, block, layers, **kwargs)
        model_keys = model.layers.keys()
        if pretrained:
            # state_dict = load_state_dict_from_url(model_urls[arch],
            #                                       progress=progress)
            if model_type == 'resnet34':
                resnet = models.resnet34(pretrained=True)
            elif model_type == 'resnet50':
                resnet = models.resnet50(pretrained=True)
            elif model_type == 'resnet101':
                resnet = models.resnet101(pretrained=True)
            elif model_type == 'resnet152':
                resnet = models.resnet152(pretrained=True)
            model.load_state_dict(resnet.state_dict())
            for k in model._modules.keys():
              if k not in model_keys:
                del model._modules[k] 
        return model
    
    def forward(self,x):
        (l0, l1, l2, l3), g = self.resnet(x)
        return l0, l1, l2, l3 ,g