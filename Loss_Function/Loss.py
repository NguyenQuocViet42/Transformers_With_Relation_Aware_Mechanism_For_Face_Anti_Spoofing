import torch.nn as nn
import torch


def Contrastive_convolution_kernel(map, index, device):
    # Define a custom 2D convolution kernel (3x3 kernel)
    custom_kernel = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=torch.float32)
    x = index // 3
    y = index % 3
    custom_kernel[x][y] = -1
    custom_kernel = custom_kernel.to(device)
    # map = map.unsqueeze(0).unsqueeze(0)
    # Reshape the kernel to match PyTorch's expected shape
    # (out_channels, in_channels, kernel_height, kernel_width)
    custom_kernel = custom_kernel.view(1, 1, 3, 3)

    # Create a custom convolutional layer using the custom kernel
    custom_conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding = 1)
    custom_conv_layer.weight.data = custom_kernel

    # Apply the custom convolution layer to the input tensor
    output_tensor = custom_conv_layer(map)
    return output_tensor[0,0,:,:]


def CRA_Loss(y_pred, y_real, depth_pred, depth_real, device,s = 1, a = 0.4):
    loss = nn.CrossEntropyLoss()
    L_Binary = loss(y_pred, y_real)
    
    L_MSE = torch.sum(torch.sqrt((depth_pred - depth_real)**2))
    
    L_CDL = 0
    list_index = [0,1,2,3,5,6,7,8]
    for index in list_index:
        pred = Contrastive_convolution_kernel(depth_pred, index, device)
        real = Contrastive_convolution_kernel(depth_real, index, device)
        L_CDL += (pred - real)**2
    L_CDL = torch.sum(torch.sqrt(L_CDL))
    
    L_Depth = L_MSE + L_CDL
    
    L_Overall = s*a*L_Binary + s*(1-a)*L_Depth

    return L_Overall