import torch
from torch.utils.data import Dataset
import numpy as np


class BeitLocalDataset(Dataset):
    def __init__(self, size):
        super(BeitLocalDataset, self).__init__()
        # self.random_sample = self._get_sample_1()
        self.random_sample = {'img': torch.rand(size=[1, 224, 224], dtype=torch.float32),
        'image_for_tokenizer': torch.rand(size=[1, 224, 224], dtype=torch.float32),
        'vis_pos': torch.randint(low=0, high=10, size=[121, 2], dtype=torch.int64),
        'vis_idx': torch.randint(low=0, high=10, size=[121], dtype=torch.int64),
        'all_idx': torch.randint(low=0, high=10, size=[196], dtype=torch.int64),
        'vis_mask': torch.randint(low=0, high=2, size=[14, 14], dtype=torch.bool),
        'invis_mask': torch.randint(low=0, high=2, size=[14, 14], dtype=torch.bool)
        }
        self._size = size

    def __getitem__(self, item):
        return self.random_sample

    def __len__(self):
        return self._size


    def _get_sample_1(self):
        img = torch.rand(size=[1, 224, 224], dtype=torch.float32)
        image_for_tokenizer = torch.rand(size=[1, 224, 224], dtype=torch.float32)
        vis_pos = torch.randint(low=0, high=10, size=[121, 2], dtype=torch.int32)
        vis_pos = np.random.randint(low=0, high=10, size=[121, 2], dtype=np.int64)
        vis_idx = np.random.randint(low=0, high=10, size=[121], dtype=np.int64)
        all_idx = np.random.randint(low=0, high=10, size=[196], dtype=np.int64)
        vis_mask = np.random.randint(low=0, high=1, size=[14, 14], dtype=np.int8).astype(np.bool8)
        invis_mask = np.random.randint(low=0, high=1, size=[14, 14], dtype=np.int8).astype(np.bool8)
        return {'img': img,
                'image_for_tokenizer': image_for_tokenizer,
                'vis_pos': vis_pos,
                'vis_idx': vis_idx,
                'all_idx': all_idx,
                'vis_mask': vis_mask,
                'invis_mask': invis_mask}
