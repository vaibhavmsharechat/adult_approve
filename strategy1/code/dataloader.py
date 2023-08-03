import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
torch.set_default_tensor_type(torch.DoubleTensor)

class MultiAvtDataLoader(Dataset):
    def __init__(self, args, train_csv1, video_feats_paths, split):
        self.args = args
        print(f"Reading {train_csv1}")
        self.csv = pd.read_csv(train_csv1)
        self.video_feats_paths = video_feats_paths

    def __len__(self):
        return len(self.csv)

    def __get_clip_feats__(self, postId):
        np_path = os.path.join(self.video_feats_paths, str(postId)+".npy")

        arr = np.load(np_path)
        
        arr = arr.astype('double')
       
        array_sum = np.sum(arr)
        array_has_nan = np.isnan(array_sum)
        if(array_has_nan):
            arr = np.zeros((512))
            arr = arr.astype('double')
        if(len(arr)!=512):
            arr = np.zeros((512))
            arr = arr.astype('double')
        norm = np.linalg.norm(arr)

        if(norm != 0 ):
            arr = arr/norm
        return arr

    def __getitem__(self, idx):
        postId = int(self.csv.iloc[idx, 0])
        target_id = int(self.csv.iloc[idx, 2])
        arr = self.__get_clip_feats__(postId)
        if arr is None:
            return None
        return {'data' : torch.from_numpy(arr),  'target' : target_id ,'id' : postId}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)