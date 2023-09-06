import torch.utils.data as data
import cv2
import numpy as np
import torch

class CRA_Dataset(data.Dataset):
    def __init__(self, data_frame, transform=None):
        super().__init__()
        self.data_frame = data_frame
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self,idx):
        path, label = self.data_frame.iloc[idx, 0], self.data_frame.iloc[idx, 1]
        
        y = [0] * 10
        y[label] = 1
        y = np.array(y)
        
        name = path.split('/')[-1]
        num_name = name.split('.')[0]
        
        # Load
        if label == 0:
            path_face = "/workspace/personal/vietnq/PRNet/face_cropped/" + num_name + '.jpg'
            path_depth_map = "/workspace/personal/vietnq/PRNet/depth_map/" + num_name + '_depth.png'
            face = cv2.imread(path_face)
            depth_map = cv2.imread(path_depth_map, cv2.IMREAD_GRAYSCALE)
            depth_map = cv2.resize(depth_map, (28, 28))
            # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        else:
            path_face = "/workspace/personal/vietnq/PRNet/spoof_cropped/" + name
            face = cv2.imread(path_face)
            depth_map = np.zeros(shape = (1, 28, 28), dtype = float)
            
        # Transform
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # face = face.astype(cv2.float) / 255.0
        if self.transform:
            sample = {
            "image": face
        }
        sample = self.transform(**sample)
        face = sample["image"]
            
        # To Tencor
        # face = torch.from_numpy(face)
        face = torch.tensor(face, dtype = torch.float32)
        # depth_map = torch.from_numpy(depth_map)
        depth_map = torch.tensor(depth_map, dtype = torch.float32)
        # y = torch.from_numpy(y)
        y = torch.tensor(y, dtype = torch.float32)
        
        return face, y, depth_map