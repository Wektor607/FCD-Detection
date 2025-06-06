import torch
from torch import nn

class Ensemble(nn.Module):
    """
    Ensemble models by taking mean of output
    Supported outputs: log_softmax, hemi_log_softmax, non_lesion_logits
    """
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        """
        Forward pass
        """
        estimates_list = []
        feature_maps_list = []

        for model in self.models:
            out = model(x)
            if isinstance(out, tuple):
                estimates, fmap = out
                estimates_list.append(estimates)
                feature_maps_list.append(fmap)
            else:
                estimates_list.append(out)
        
        ensembled_estimates = {}
        for key in ['log_softmax', 'hemi_log_softmax', 'non_lesion_logits','object_detection_linear']:
            if key not in estimates_list[0].keys():
                continue
            if 'log_softmax' in key:
                # there are the logged outputs -> before mean, need to do exp 
                vals = torch.stack([torch.exp(est[key]) for est in estimates_list], dim=2)
                mean_val = torch.log(torch.mean(vals, dim=2))
                #testing out max instead of mean
                #mean_val = torch.log(torch.max(vals, dim=2)[0])
            else:
                mean_val = torch.mean(torch.stack([est[key] for est in estimates_list], dim=2), dim=2)
            ensembled_estimates[key] = mean_val
        return ensembled_estimates, feature_maps_list