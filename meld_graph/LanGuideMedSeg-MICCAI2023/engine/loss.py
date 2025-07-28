import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import torch
from functools import partial

def dice_coeff(pred, target, smooth=1e-15):
    """
    Differentiable Dice coefficient implementation.

    This definition generalizes to real valued pred and target vector.
    NOTE assumes that pred is softmax output of model, might need torch.exp on pred before.

    Args:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch (not one-hot encoded)
    """
    # make target one-hot encoded (also works for soft targets)
    target_hot = torch.transpose(torch.stack((1 - target, target)), 0, 1)
    iflat = pred.contiguous()
    tflat = target_hot.contiguous()
    intersection = (iflat * tflat).sum(dim=0)
    A_sum = torch.sum(iflat * iflat, dim=0)
    B_sum = torch.sum(tflat * tflat, dim=0)
    dice = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)
    return dice


class DiceLoss(torch.nn.Module):
    """
    Dice loss.

    Args:
        loss_weight_dictionary (dict): loss dict from experiment_config.py
    """

    def __init__(self, loss_weight_dictionary=None):
        super(DiceLoss, self).__init__()
        self.class_weights = [0.0, 1.0]
        if "dice" in loss_weight_dictionary.keys():
            if "class_weights" in loss_weight_dictionary["dice"]:
                self.class_weights = loss_weight_dictionary["dice"]["class_weights"]
            if "epsilon" in loss_weight_dictionary["dice"]:
                self.epsilon = loss_weight_dictionary["dice"]["epsilon"]
            else:
                self.epsilon = 1e-15

    def forward(self, inputs, targets, device=None, **kwargs):
        dice = dice_coeff(torch.exp(inputs), targets, smooth=self.epsilon)
        class_weights = torch.tensor(self.class_weights, dtype=float)
        if device is not None:
            class_weights = class_weights.to(device)

        dice = dice[0] * class_weights[0] + dice[1] * class_weights[1]
        return 1 - dice


class CrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss (NLLLoss).
    """

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        return self.loss(inputs, targets)


class SoftCrossEntropyLoss(torch.nn.Module):
    """
    Soft version of cross entropy loss.
    Equivalent to CE if labels/targets are hard.
    """

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, do not need to log
        # formula: non-lesional (inputs[:0]) + lesional (inputs[:1])
        ce = -(1 - targets) * inputs[:, 0] - targets * inputs[:, 1]
        return torch.mean(ce)


class MAELoss(torch.nn.Module):
    """
    L1 loss.
    """

    def __init__(self, weight=None, size_average=True):
        super(MAELoss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        return self.loss(inputs, targets)

class SmoothL1Loss(torch.nn.Module):
    """
    Smooth L1 loss for object detection. includes mask
    """
    def __init__(self, weight=None, size_average=True):
        super(SmoothL1Loss, self).__init__()
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, inputs, targets,xyzr, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        # mask out non-lesional examples from the loss using targets  
        xyzr_reshaped = xyzr.view(-1,4)
        inputs = inputs * targets.unsqueeze(1).float()
        xyzr_reshaped = xyzr_reshaped * targets.unsqueeze(1).float()
        loss = self.loss(inputs, xyzr_reshaped)
       # masked_loss = loss * targets.float()  # Apply binary mask
        return loss
    

class DistanceRegressionLoss(torch.nn.Module):
    """
    Distance regression loss. Either MSE, MAE, MLE
    Args:
        params (dict): loss dict from experiment_config.py
    """

    def __init__(self, params):
        super(DistanceRegressionLoss, self).__init__()
        if "distance_regression" in params.keys():
            self.weigh_by_gt = params["distance_regression"].get("weigh_by_gt", False)
            self.loss = params["distance_regression"].get("loss", "mse")
            assert self.loss in ["mse", "mae", "mle"]
        else:
            self.weigh_by_gt = False
            self.loss='mse'
    
    def forward(self, inputs, target, distance_map, **kwargs):
        inputs = torch.squeeze(inputs)
        # normalise distance map
        distance_map = torch.div(distance_map, 300)
        # calculate loss
        if self.loss == "mse":
            loss = torch.square(torch.subtract(inputs, distance_map))
        elif self.loss == "mae":
            loss = torch.abs(torch.subtract(inputs, distance_map))
        elif self.loss == "mle":
            loss = torch.log(torch.add(torch.abs(torch.subtract(inputs, distance_map)), 1))
        # weigh loss
        if self.weigh_by_gt:
            loss = torch.div(loss, torch.add(distance_map, 1))
        loss = loss.mean()
        
        return loss


class FocalLoss(torch.nn.Module):
    """
    Focal loss.
    """

    def __init__(self, params, size_average=True):
        super(FocalLoss, self).__init__()
        try:
            self.gamma = params["focal_loss"]["gamma"]
        except:
            self.gamma = 0
        try:
            self.alpha = params["focal_loss"]["alpha"]
        except:
            self.alpha = None
        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])
        self.size_average = size_average

    def forward(self, inputs, target, **kwargs):
        target = target.view(-1, 1)
        
        logpt = inputs
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def get_sensitivity(pred, target):
    """
    Sample-level sensitivity.
    Returns 1 if any TP, 0 otherwise.
    """
    if torch.sum(torch.logical_and((target == 1), (pred == 1))) > 0:
        return 1
    else:
        return 0


def tp_fp_fn_tn(pred, target):
    """
    Returns TP, FP, FN, TN.
    """
    tp = torch.sum(torch.logical_and((target == 1), (pred == 1)))
    fp = torch.sum(torch.logical_and((target == 0), (pred == 1)))
    fn = torch.sum(torch.logical_and((target == 1), (pred == 0)))
    tn = torch.sum(torch.logical_and((target == 0), (pred == 0)))
    return tp, fp, fn, tn
    

def calculate_loss(loss_dict, estimates_dict, labels,
                    distance_map=None, xyzr=None,
                    deep_supervision_level=None, device=None, n_vertices=None):
    """ 
    Calculate loss. Can combine losses with weights defined in loss_dict

    Example loss_dict:
    ```
    loss_dict = {
        'cross_entropy':{'weight':1},
        'focal_loss':{'weight':1, 'alpha':0.4, 'gamma':4},
        'dice':{'weight': 1, 'class_weights': [0.0, 1.0]},
        'distance_regression': {'weight': 1, 'weigh_by_gt': True},
        'lesion_classification': {'weight': 1, 'apply_to_bottleneck': True}
        }
    ```
    !!!!!! NOTE Estimates are the ``logSoftmax`` output of the model. For some losses, applying torch.exp is necessary! !!!!!!

    Args:
        loss_dict (dict): define losses that should be caluclated.
        estimates_dict (dict): model outputs dictionary. 
        labels (tensor): groundtruth lesion labels.
        distance_map (optional, tensor): groundtruth distance map. 
        xyzr (optional, tensor): groundtruth xyzr coordinates for object detection
        deep_supervision_level (optional, int): calculate_loss is called for every deep supervision level. 
            This arg indicates which level we are currenly at.
            Used to get the correct outputs from estimates_dict.
        n_vertices: number of vertices at current level.
    """
    loss_functions = {
        'dice': partial(DiceLoss(loss_weight_dictionary=loss_dict),
                         device=device),
        'cross_entropy': CrossEntropyLoss(),
        'soft_cross_entropy': SoftCrossEntropyLoss(),
        'focal_loss': FocalLoss(loss_dict),
        'distance_regression': DistanceRegressionLoss(loss_dict),
        'lesion_classification': CrossEntropyLoss(),
        'mae_loss': MAELoss(),
        'object_detection': SmoothL1Loss()
    }
    if distance_map is not None:
        distance_map.to(device)
    losses = {}
    for loss_def in loss_dict.keys():
        # TODO if deep supverision level
        # Return later
        prefix = ""
        # if deep_supervision_level is None:
        #     prefix = ""
        # else:
        #     prefix = f"ds{deep_supervision_level}_"

        cur_labels = labels
        if loss_def in ['dice', 'cross_entropy', 'focal_loss','mae_loss','soft_cross_entropy']:
            cur_estimates = estimates_dict[f'{prefix}log_softmax']
        elif loss_def == 'distance_regression':
            cur_estimates = estimates_dict[f'{prefix}non_lesion_logits']
        elif loss_def == 'object_detection':
            #object detection only on bottleneck. pass current labels for classification mask
            if deep_supervision_level is not None:
                continue
            else:
                cur_estimates = estimates_dict['object_detection_linear']
                cur_labels = torch.any(labels.view(labels.shape[0]//n_vertices, -1), dim=1).long()
        elif loss_def == 'lesion_classification':
            if loss_dict[loss_def].get('apply_to_bottleneck', False):
                # if apply lc to bottleneck, do not apply it on deep supervision levels
                if deep_supervision_level is not None:
                    continue
                else:
                    # on highest level, can apply lc
                    cur_estimates = estimates_dict["hemi_log_softmax"]
            else:
                cur_estimates = estimates_dict[f"{prefix}log_sumexp"]
            cur_labels = torch.any(labels.view(labels.shape[0] // n_vertices, -1), dim=1).long()
        else:
            raise NotImplementedError(f"Unknown loss def {loss_def}")

        losses[loss_def] = loss_dict[loss_def]['weight'] * loss_functions[loss_def](cur_estimates, cur_labels, 
                                                                                    distance_map=distance_map,xyzr=xyzr)
    return losses
