import os
import sys
sys.path.append(os.getcwd()+'/model')

import Layer
import torch
import torch.nn as nn

class ViT_CRA(nn.Module):
    C = 192
    def __init__(self):
        super(ViT_CRA, self).__init__()
        self.input_layer = Layer.Input_layer()
        # Encoder Layers
        self.encoder_layers = Layer.Encoder_layer()
        # self.encoder_layer_2 = Layer.Encoder_layer()
        # self.encoder_layer_3 = Layer.Encoder_layer()
        # self.encoder_layer_4 = Layer.Encoder_layer()
        # self.encoder_layer_5 = Layer.Encoder_layer()
        # self.encoder_layer_6 = Layer.Encoder_layer()
        # self.encoder_layer_7 = Layer.Encoder_layer()
        # self.encoder_layer_8 = Layer.Encoder_layer()
        # self.encoder_layer_9 = Layer.Encoder_layer()
        # self.encoder_layer_10 = Layer.Encoder_layer()
        # self.encoder_layer_11 = Layer.Encoder_layer()
        # self.encoder_layer_12 = Layer.Encoder_layer()
        # Batchnorm layer
        # self.batchnorm_layer_1 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_2 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_3 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_4 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_5 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_6 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_7 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_8 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_9 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_10 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_11 = nn.BatchNorm2d(self.C)
        # self.batchnorm_layer_12 = nn.BatchNorm2d(self.C)
        # CRA Layers
        self.cra_layer_36 = Layer.CRA()
        self.cra_layer_69 = Layer.CRA()
        self.cra_layer_912 = Layer.CRA() 
        # Decoder Layer
        self.decoder_layer = Layer.Decoder_layer()
        # MLP Layer
        self.mlp_layer = Layer.MLP()
    
    def forward(self, image):
        # Linear Projection
        x = self.input_layer(image)
        
        # Encoder
        list_ouput = self.encoder_layers(x)
        x_encoder_1 = list_ouput[0]
        # x_encoder_1 = self.batchnorm_layer_1(x_encoder_1)
        
        x_encoder_2 = list_ouput[1]
        # x_encoder_2 = self.batchnorm_layer_2(x_encoder_2)
        
        x_encoder_3 = list_ouput[2]
        # x_encoder_3 = self.batchnorm_layer_3(x_encoder_3)
        
        x_encoder_4 = list_ouput[3]
        # x_encoder_4 = self.batchnorm_layer_4(x_encoder_4)
        
        x_encoder_5 = list_ouput[4]
        # x_encoder_5 = self.batchnorm_layer_5(x_encoder_5)
        
        x_encoder_6 = list_ouput[5]
        # x_encoder_6 = self.batchnorm_layer_6(x_encoder_6)
        
        x_encoder_7 = list_ouput[6]
        # x_encoder_7 = self.batchnorm_layer_7(x_encoder_7)
        
        x_encoder_8 = list_ouput[7]
        # x_encoder_8 = self.batchnorm_layer_8(x_encoder_8)
        
        x_encoder_9 = list_ouput[8]
        # x_encoder_9 = self.batchnorm_layer_9(x_encoder_9)
        
        x_encoder_10 = list_ouput[9]
        # x_encoder_10 = self.batchnorm_layer_10(x_encoder_10)
        
        x_encoder_11 = list_ouput[10]
        # x_encoder_11 = self.batchnorm_layer_11(x_encoder_11)
        
        x_encoder_12 = list_ouput[11]
        # x_encoder_12 = self.batchnorm_layer_12(x_encoder_12)
        
        # MLP
        x_mlp = x_encoder_12[:,0,:]
        y_mlp = self.mlp_layer(x_mlp)
        
        # CRA
        x_encoder_912 = self.cra_layer_912(x_encoder_9[:,1:,:], x_encoder_12[:,1:,:])
        x_encoder_912 = x_encoder_912.permute(0, 2, 1)
        
        x_encoder_69 = self.cra_layer_69(x_encoder_6[:,1:,:], x_encoder_912)
        x_encoder_69 = x_encoder_69.permute(0, 2, 1)
        
        x_encoder_36 = self.cra_layer_36(x_encoder_3[:,1:,:], x_encoder_69)
        x_encoder_36 = x_encoder_36.permute(0, 2, 1)
        # Decoder
        x_decoder = x_encoder_36.reshape(x_encoder_36.shape[0], 14, 14, 192)
        x_decoder = x_decoder.permute(0, 3, 1, 2)
        depth_map = self.decoder_layer(x_decoder)
        
        return y_mlp, depth_map