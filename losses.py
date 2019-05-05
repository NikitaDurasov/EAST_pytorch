import torch
import numpy as np
import datasets


class BalancedXent(torch.nn.Module):

    def __call__(self, prediction, ground, mask):

        positive_pixels_sum = ground.sum(dim=[1, 2, 3]).float()

        beta = 1 - positive_pixels_sum / np.prod(ground.shape[2:])
        beta = beta.reshape([beta.shape[0], 1, 1, 1]).type_as(prediction).float()

        result = beta * ground * torch.log(prediction + 1e-4) + \
            (1 - beta) * (1 - ground) * torch.log(1 - prediction + 1e-4)

        return -result[mask].mean()


class QuadGeometry(torch.nn.Module):

    def __call__(self, pred_geom, gt_geom, mask):
        eps = 1e-4        
        
        predicted_geom = pred_geom.permute(0, 2, 3, 1)[mask.squeeze(1).byte()]
        ground_geom = gt_geom.permute(0, 2, 3, 1)[mask.squeeze(1).byte()]
        
#         print()
#         print("PREDICTED GEOM FULL - min: {0}, mean: {1}, max: {2}".format(pred_geom.min(), 
#                                                                            pred_geom.mean(), 
#                                                                            pred_geom.max()))
#         print("GROUND GEOM FULL - min: {0}, mean: {1}, max: {2}".format(gt_geom.min(), 
#                                                                         gt_geom.mean(), 
#                                                                         gt_geom.max()))
#         print("GROUND GEOM - min: {0}, mean: {1}, max: {2}".format(ground_geom.min(), 
#                                                                    ground_geom.mean(), 
#                                                                    ground_geom.max()))
#         print("PREDICTED GEOM - min: {0}, mean: {1}, max: {2}".format(predicted_geom.min(), 
#                                                                       predicted_geom.mean(), 
#                                                                       predicted_geom.max()))
        
        
        smooth_loss = torch.nn.functional.smooth_l1_loss(predicted_geom,
                                                         ground_geom,
                                                         reduction='mean')

        if torch.isnan(smooth_loss):
            return None

#         predicted_bboxes = predicted_geom.permute(0, 2, 3, 1)[mask.squeeze(1).byte()]

        normalization = datasets.dataset_utils.short_length_bbox_size(
            predicted_geom.cpu().detach().numpy())

        normalization = torch.from_numpy(normalization).type_as(predicted_geom)
        normalization = normalization.reshape(-1, 1)

        return (smooth_loss / (normalization + 1)).mean() 

class RboxGeometry(torch.nn.Module):
    pass
