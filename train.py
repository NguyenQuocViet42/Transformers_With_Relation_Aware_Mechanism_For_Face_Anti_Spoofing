from model.ViT_CRA import ViT_CRA
from Dataset.MyDataset import CRA_Dataset
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.optim import lr_scheduler
from Dataset.Transform import train_transform
from Dataset.Transform import test_transform
from Loss_Function.Loss import CRA_Loss
from model.ViT_CRA import ViT_CRA

def set_random_seeds(random_seed=0):
    
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
set_random_seeds(42)

df = pd.read_csv('annotation/depth_train_annotation.csv')
df = df.iloc[:2000]

# Split Training Set and Validation Set
num_samples = df.shape[0]
num_samples_train_set = int(num_samples * 0.8)

random_indices = np.random.permutation(num_samples)

train_set_indices = random_indices[:num_samples_train_set]
train_set = df.iloc[train_set_indices]

val_set_indices = random_indices[num_samples_train_set:]
val_set = df.iloc[val_set_indices]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid Function
def validation(pred, depth_map,y_val, a = 0.4, threshold = 0.5):
    size = len(pred)
    count = 0
    for i in range(size):
        # prediction =  torch.argmax(pred[i])
        live_logit = pred[0]
        Dmean = torch.sum(depth_map) / (28*28)
        # if prediction != 0:
        #     prediction = 1
        real = torch.argmax(y_val[i])
        # if real != 0:
        #     real = 1
        Score = a*live_logit + (1-a)*Dmean
        if Score > threshold:
            prediction = 0
        else:
            prediction = 1
        if real == prediction:
            count += 1
    return count / size * 100

def write_matrix_to_txt(vector, filename):
    try:
        with open(filename, 'w') as file:
            vector_str = ' '.join(str(element) for element in vector)
            file.write(vector_str)
        print("Ma trận 1 chiều đã được ghi vào file thành công.")
    except Exception as e:
        print("Đã xảy ra lỗi:", e)   

best_val_loss = float(99)
best_val_acc = [float(0)] * 4

train_losses = []
val_acces = []
val_losses = []

def train(epochs):
    global train_losses, val_acces, val_losses
    global best_val_acc, best_val_loss
    # optimizer
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = CRA_Loss
    model = ViT_CRA()
    
    train_dataset = CRA_Dataset(train_set, transform = train_transform)
    val_dataset = CRA_Dataset(val_set, transform = test_transform)

    torch.distributed.init_process_group(backend="nccl")
    train_sampler = DistributedSampler(dataset=train_dataset)

    train_dataloader = data.DataLoader(train_dataset, batch_size = 2, sampler = train_sampler, num_workers = 4)
    valid_dataloader = data.DataLoader(val_dataset, batch_size = 2, num_workers = 4)
    

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="Local rank. Necessary for using the torch.distributed.launch utility.")
    argv = parser.parse_args()
    local_rank = argv.local_rank
    print("Rank: ", local_rank)
    
    device = torch.device("cuda:{}".format(local_rank))
    model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    
    lr = 0.01
    optimizer = optim.Adam(ddp_model.parameters(), lr = lr, weight_decay=0.00005)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 45], gamma=0.1)           #
    
    for epoch in range(epochs):
        # lr_schedule.print_lr()
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        ddp_model.train()
        loss = 0
        val_acc = 0
        
        for x_batch,y_batch, map_batch in train_dataloader:
            # torch.autograd.set_detect_anomaly(True)
            x_train, y_train, map_train = x_batch, y_batch, map_batch
            # getting the validation set
            #xóa gradients
            optimizer.zero_grad()
            # Cho dữ liệu qua model và trả về output cần tìm
            pred, depth_map = ddp_model(x_train.to(device))
            # Tính toán giá trị lỗi và backpropagation
            loss = criterion(pred, y_train.to(device), depth_map, map_train.to(device), device)
            loss.backward()
            # torch.distributed.barrier()
            # Cập nhật trọng số
            optimizer.step()
            torch.autograd.set_detect_anomaly(False)
        lr_schedule.step()
        train_losses.append(loss.item())
        
        #Thiết lập trạng thái đánh giá cho mô hình, ở bước này thì mô hình không backward và cập nhật trọng số
        ddp_model.eval()
        for x_batch,y_batch, map_batch in valid_dataloader:
            x_val, y_val, map_val = x_batch, y_batch, map_batch
            pred, depth_map = ddp_model(x_val.to(device))
            val_loss = criterion(pred, y_val.to(device), depth_map, map_val.to(device), device)
            val_acc = validation(pred, depth_map, y_val.to(device))
        if val_acc > best_val_acc[local_rank]:
            best_val_acc[local_rank] = val_acc
            torch.save(ddp_model.state_dict(), '../best_check_point/Epoch_' + str(epoch) +'_GPU_'+ str(local_rank) + '_best_check_point.pth')
            print(str(local_rank) + '_Best val_acc: ', val_acc)
            
        elif val_acc == best_val_acc[local_rank] and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc[local_rank] = val_acc
            torch.save(ddp_model.state_dict(), '../best_check_point/Epoch_' + str(epoch) +'_GPU_'+ str(local_rank) + '_best_check_point.pth')
            print(str(local_rank) + '_Best val_acc: ', val_acc)

        torch.save(ddp_model.state_dict(), '../check_point/Epoch_' + str(epoch) +'_GPU_'+ str(local_rank) + '_best_check_point.pth')
        
        val_acces.append(val_acc)
        val_losses.append(val_loss.item())
        print('Epoch : ',epoch+1, '\t', 'loss :', loss.item(), '\t', 'val_loss :', val_loss.item(),  '\t', 'Val_acc :', val_acc, '\n') 
        
        # count += 1 
        # if count == 15:
        #     count = 0
        #     lr = lr * 0.1
        #     optimizer = optim.Adam(ddp_model.parameters(), lr = lr, weight_decay=0.0001)
    write_matrix_to_txt(train_losses, 'train_loss/' + str(local_rank) + '_train_loss.txt')
    write_matrix_to_txt(val_losses, 'val_loss/' + str(local_rank) + '_val_loss.txt')
    write_matrix_to_txt(val_acces, 'val_accuracy/' + str(local_rank) + '_val_accuracy.txt')
    
if __name__ == "__main__":
    
    train(60)
    print('Best_val_accuracy: ', best_val_acc)