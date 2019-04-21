from torch.utils.data import Dataset
from . import dataset_utils
from PIL import Image
import numpy as np

import os

class DebugDataset(Dataset):

    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.datafiles = self.generate_files()

    def generate_files(self):
        gt_path = os.path.join(self.data_path, self.split, 'gt')
        img_path = os.path.join(self.data_path, self.split, 'img')

        id_range = range(1, len(os.listdir(img_path)) + 1)
        img_files = ["img_{}.jpg".format(idx) for idx in id_range]
        gt_files = ["gt_{}.txt".format(idx) for idx in id_range]

        img_files = [os.path.join(img_path, img) for img in img_files]
        gt_files = [os.path.join(gt_path, gt) for gt in gt_files]

        return list(zip(img_files, gt_files))

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        img_file, gt_file = self.datafiles[index]
        img = np.array(Image.open(img_file))
        bboxes = dataset_utils.read_gt(gt_file)

        sample = {"image": img, "bboxes": bboxes}

        if self.transform:
            sample = self.transform(sample)

        return sample
