import torch.nn as nn
import torch
from einops.layers.torch import Rearrange


class Input_layer(nn.Module):
    patch_size = 16
    num_patches = (224 // patch_size)**2 + 1
    embed_dim = 196
    
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
        print(Zpos.shape)
        output = Zcls_p + Zpos
        return output

class Encoder_layer(nn.Module):
    def __init__(self):
        super(Encoder_layer, self).__init__()
