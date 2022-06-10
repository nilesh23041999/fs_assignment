import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor,self).__init__()
        self.newmodel = nn.Sequential(*(list(self.model.children())[:-1]))
        
    def forward(self,x):
        x = self.newmodel(x)
        x = x.squeeze()
        #x = x.reshape((x.shape[0], -1))
        print("Feature",x.shape)
        return x


