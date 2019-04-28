import torch
import numpy as np
import datasets


class BalancedXent(torch.nn.Module):

    def __call__(self, prediction, ground, mask):

        positive_pixels_sum = ground.sum(dim=[1, 2, 3]).float()

        beta = 1 - positive_pixels_sum / np.prod(ground.shape[2:])
        beta = beta.reshape([beta.shape[0], 1, 1, 1]).float()

        result = beta * ground * torch.log(prediction + 1e-3) + \
            (1 - beta) * (1 - ground) * torch.log(1 - prediction + 1e-3)

        return -result[mask].mean()


class QuadGeometry(torch.nn.Module):

    def __call__(self, predicted_geom, ground_geom, mask):
        eps = 1e-4

        smooth_loss = torch.nn.functional.smooth_l1_loss(predicted_geom,
                                                         ground_geom,
                                                         reduction='none')
        geometry_loss = smooth_loss.permute(0, 2, 3, 1)[
            mask.squeeze(1).byte()]

        if len(geometry_loss) == 0:
            return None

        predicted_bboxes = predicted_geom.permute(0, 2, 3, 1)[mask.squeeze(1).byte()]

        normalization = datasets.dataset_utils.short_length_bbox_size(
            predicted_bboxes.detach().numpy())

        normalization = torch.from_numpy(normalization)
        normalization = normalization.reshape(-1, 1)

        result = (geometry_loss / (normalization + eps)).mean()
        if torch.isnan(result):
            return None
        return result


class RboxGeometry(torch.nn.Module):
    pass
