import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from transformers import ViTModel
import torch.nn.functional as F


class Input_layer(nn.Module):
    patch_size = 16
    num_patches = (224 // patch_size)**2 + 1
    embed_dim = 768
    
    def flatten_image(self, batch_image):
        output = []
        for i in range(batch_image.shape[0]):
            image = batch_image[i]
            image = image.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
            image = image.reshape(-1, 3, self.patch_size, self.patch_size)
            image = image.reshape(image.size(0),-1)
            output.append(image)
        return torch.stack(output)
    
    def __init__(self):
        super(Input_layer, self).__init__()
        
        
        self.patch_embedding=nn.Sequential(
            # Rearrange('b c (h px) (w py) -> b (h w) (px py c)', px=self.patch_size, py=self.patch_size),
            nn.Linear(self.patch_size*self.patch_size*3, self.embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_embedding=nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
    
    def forward(self, batch_image):
        patches = self.flatten_image(batch_image)
        Zp = self.patch_embedding(patches)
        Zcls = self.cls_token.repeat(batch_image.size(0), 1, 1)
        Zcls_p = torch.cat((Zcls, Zp), dim=1)
        Zpos = self.positional_embedding
        output = Zcls_p + Zpos
        return output

class Encoder_layer(nn.Module):
    def __init__(self):
        super(Encoder_layer, self).__init__()
        model_trans = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        list_layer = list(model_trans.children())
        self.encoder_layer = nn.Sequential(list_layer[1])
        
    def forward(self, x):
        x = self.encoder_layer(x)
        return x.last_hidden_state
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 4096)
        self.norm_1 = nn.BatchNorm1d(4096)
        self.gelu_1 = F.gelu()
        self.dr_1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.norm_2 = nn.BatchNorm1d(4096)
        self.gelu_2 = F.gelu()
        self.dr_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 10)

        
    def foward(self, x):
        x = self.fc1(x)
        x = self.norm_1(x)
        x = self.gelu_1(x)
        x = self.dr_1(x)
        x = self.fc2(x)
        x = self.norm_2(x)
        x = self.gelu_2(x)
        x = self.dr_2(x)
        x = self.fc3(x)
        return x
