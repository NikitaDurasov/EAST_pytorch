import torch
import numpy as np
import datasets

class BalancedXent(torch.nn.Module):

    def __call__(self, prediction, ground):
        positive_pixels_sum = ground.sum(dim=[1, 2, 3]).float()
        beta = 1 - positive_pixels_sum / np.prod(ground.shape[2:])
        beta = beta.reshape([beta.shape[0], 1, 1, 1])

        result = beta * ground * torch.log(prediction) + \
            (1 - beta) * (1 - ground) * torch.log(1 - prediction)

        return result.mean()


class QuadGeometry(torch.nn.Module):

    def __call__(self, predicted_bboxes, ground_bboxes):
        smooth_loss = torch.nn.functional.smooth_l1_loss(predicted_bboxes,
                                                         ground_bboxes,
                                                         reduction='none')
        print(smooth_loss)
        normalization = datasets.dataset_utils.short_length_bbox_size(
            predicted_bboxes.detach().numpy())
        print(normalization)
        normalization = torch.from_numpy(normalization)
        normalization = normalization.reshape(-1, 1)
        return (smooth_loss / normalization).mean()

class RboxGeometry(torch.nn.Module):
    pass