import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from transformers import ViTModel, DeiTForImageClassificationWithTeacher
import torch.nn.functional as F

class Input_layer(nn.Module):
    patch_size = 16
    num_patches = (224 // patch_size)**2 + 1
    embed_dim = 192
    
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
        
        
        self.patch_embedding = nn.Sequential(
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
        model_trans = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
        list_layer = list(model_trans.children())
        list_layer = list(list_layer[0].children())[1]
        list_layer = list(list(list_layer.children())[0].children())
        self.list_secquence = [nn.Sequential(i) for i in list_layer] # list 12 Encoder layers
        
    def forward(self, x):
        list_output = []
        for i in range(12):
            x = self.list_secquence[i](x)
            list_output.append(x.last_hidden_state)
        return list_output
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(192, 4096)
        self.norm_1 = nn.BatchNorm1d(4096)
        self.dr_1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.norm_2 = nn.BatchNorm1d(4096)
        self.dr_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 10)

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm_1(x)
        x = F.gelu(x)
        x = self.dr_1(x)
        x = self.fc2(x)
        x = self.norm_2(x)
        x = F.gelu(x)
        x = self.dr_2(x)
        x = self.fc3(x)
        return x

class Decoder_layer(nn.Module):
    in_channels = 192
    out_channels = 1
    
    def __init__(self):
        super(Decoder_layer, self).__init__()
        
        # Convolutional layers for upsampling
        self.conv1 = nn.ConvTranspose2d(self.in_channels, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        
        # Normalization layers
        self.norm1 = nn.BatchNorm2d(512)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(64)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Final convolutional layer to get the depth map
        self.final_conv_1 = nn.Conv2d(128, 64, kernel_size=2, stride=2)
        self.final_conv_2 = nn.Conv2d(64, self.out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.final_conv_1(x)
        x = self.norm4(x)
        x = self.relu(x)
        depth_map = self.final_conv_2(x)
        return depth_map
    
class CRA(nn.Module):
    C = 192
    def __init__(self):
        super( CRA, self).__init__()
        self.linear_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels= self.C, out_channels = self.C, kernel_size= 1),
            nn.BatchNorm1d(self.C),
            nn.ReLU(),
        )
        self.linear_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels= self.C, out_channels = self.C, kernel_size= 1),
            nn.BatchNorm1d(self.C),
            nn.ReLU(),
        )
        self.linear_conv_3 = nn.Sequential(
            nn.Conv1d(in_channels= self.C, out_channels = self.C, kernel_size= 1),
            # nn.BatchNorm1d(self.C),
            # nn.ReLU(),
        )
        self.linear_conv_4 = nn.Sequential(
            nn.Conv1d(in_channels = 784, out_channels = 784, kernel_size= 1),
            # nn.BatchNorm1d(self.C),
            # nn.ReLU(),
        )
        self.linear_conv_5 = nn.Sequential(
            nn.Conv1d(in_channels= self.C + 784, out_channels = 1, kernel_size= 1),
            # nn.BatchNorm1d(self.C),
            # nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        x = x.permute(0, 2, 1)          # shape = batch x C x 196
        y = y.permute(0, 2, 1)          # shape = batch x C x 196
        phi_x = self.linear_conv_1(x)   # shape = batch C x 196
        phi_y = self.linear_conv_2(y)   # shape = batch C x 196
        cat_phi = torch.cat((phi_x, phi_y), dim = 2)    # shape = batch C x 392
        
        A_arr = []
        for batch in range(phi_x.shape[0]):
            A_arr.append(torch.mm(cat_phi[batch].t(), cat_phi[batch]))
        A = torch.stack(A_arr)          # shape = batch x 392 x 392
        
        R_batch = []
        for batch in range(A.shape[0]):
            R_arr = []
            for i in range(392):
                R_arr.append( torch.cat( (A[batch, i, :], A[batch, :, i]), dim = 0 ) )
            R_batch.append(torch.stack(R_arr))
        R = torch.stack(R_batch)            # shape = batch x 392 x 784
        R = R.permute(0, 2, 1)              # shape = batch x 784 x 392
        
        v = self.linear_conv_3(cat_phi)     # shape = batch x C x 392
        v_ = self.linear_conv_4(R)          # shape = batch x 784 + 392
        cat_v = torch.cat((v, v_), dim = 1) # shape = batch x (784 + C) x 392
        W = self.linear_conv_5(cat_v)       # shape = batch x 1 x 392
        W_0 = W[:, :, :196]
        W_1 = W[:, :, 196:]
        
        # for batch in range(x.shape[0]):
        #     for i in range(196):
        #         x[batch, :, i] = x[batch, :, i] * W_0[batch, 0, i]
        #         y[batch, :, i] = y[batch, :, i] * W_1[batch, 0, i]
        
        output = x * W_0 + y * W_1
        return output           # shape = batch x C x 392