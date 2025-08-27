import torch.nn as nn
import torch
import torch.nn.functional as F
from nnunet.utilities.tensor_utilities import sum_tensor
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target):
        num = target.size(0)
        probs = torch.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = - torch.log(score.sum() / num)
        return score


def topk_loss_with_logits(predictions, targets, percentage=0.1):
    loss_tensor = F.binary_cross_entropy_with_logits(predictions, targets)
    flattened_loss = loss_tensor.view(1, -1)
    k = max(1, int(flattened_loss.size(1) * percentage))
    topk_losses, _ = torch.topk(flattened_loss, k)
    return topk_losses.mean()


def distance_field(img):
    field = np.zeros_like(img)
    for batch in range(len(img)):
        fg_mask = img[batch] > 0.5
        if fg_mask.any():
            bg_mask = ~fg_mask
            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)
            field[batch] = fg_dist + bg_dist
    return field

def hausdorff_loss_with_logits(pred, target, alpha=2):
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (pred.dim() == target.dim()), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(distance_field(pred.detach().cpu().numpy())).float().to(pred.device)
        target_dt = torch.from_numpy(distance_field(target.detach().cpu().numpy())).float().to(pred.device)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** alpha + target_dt ** alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()
        return loss