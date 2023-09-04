import os
import sys
sys.path.append(os.getcwd()+'/model')

import Layer
import torch
import torch.nn as nn

class ViT_CRA(nn.Module):
    def __init__(self):
        super(ViT_CRA, self).__init__()
        
        self.input_layer = Layer.Input_layer()
        # Encoder Layers
        self.encoder_layer_1 = Layer.Encoder_layer()
        self.encoder_layer_2 = Layer.Encoder_layer()
        self.encoder_layer_3 = Layer.Encoder_layer()
        self.encoder_layer_4 = Layer.Encoder_layer()
        self.encoder_layer_5 = Layer.Encoder_layer()
        self.encoder_layer_6 = Layer.Encoder_layer()
        self.encoder_layer_7 = Layer.Encoder_layer()
        self.encoder_layer_8 = Layer.Encoder_layer()
        self.encoder_layer_9 = Layer.Encoder_layer()
        self.encoder_layer_10 = Layer.Encoder_layer()
        self.encoder_layer_11 = Layer.Encoder_layer()
        self.encoder_layer_12 = Layer.Encoder_layer()
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
        x_encoder_1 = self.encoder_layer_1(x)
        x_encoder_2 = self.encoder_layer_2(x_encoder_1)
        x_encoder_3 = self.encoder_layer_3(x_encoder_2)
        x_encoder_4 = self.encoder_layer_4(x_encoder_3)
        x_encoder_5 = self.encoder_layer_5(x_encoder_4)
        x_encoder_6 = self.encoder_layer_6(x_encoder_5)
        x_encoder_7 = self.encoder_layer_7(x_encoder_6)
        x_encoder_8 = self.encoder_layer_8(x_encoder_7)
        x_encoder_9 = self.encoder_layer_9(x_encoder_8)
        x_encoder_10 = self.encoder_layer_10(x_encoder_9)
        x_encoder_11 = self.encoder_layer_11(x_encoder_10)
        x_encoder_12 = self.encoder_layer_12(x_encoder_11)
        
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
        x_decoder = x_encoder_36.reshape(x_encoder_36.shape[0], 14, 14, 768)
        x_decoder = x_decoder.permute(0, 3, 1, 2)
        depth_map = self.decoder_layer(x_decoder)
        
        return y_mlp, depth_map