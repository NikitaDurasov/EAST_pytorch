from . import east
import numpy as np
import torch

class BoxesPredictor:

    def __init__(self, thr=0.5):
        self.thr = thr

    def __call__(self, prediction):

        score_maps = prediction["score_map"].clone()
        geometries = prediction["geometry"].clone()
        batch_boxes = []

        for i in range(len(score_maps)):
            score_map = score_maps[i].squeeze()
            geometry = geometries[i].squeeze().permute(1, 2, 0)

            # preprocess quad format
            ones = np.ones(list(geometry.shape[:2]) + [4])
            x_coords = np.cumsum(ones, axis=1)
            y_coords = np.cumsum(ones, axis=0)

            geometry[:, :, ::2] += torch.from_numpy(x_coords).float()
            geometry[:, :, 1::2] += torch.from_numpy(y_coords).float()

            mask = score_map > self.thr
            boxes = geometry[mask]

            batch_boxes.append(boxes)

        return batch_boxes



