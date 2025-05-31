import torch
import torch.nn.functional as F

class LossFactory:
    @staticmethod
    def create_loss(config):
        loss_name = config['loss']['name']
        loss_params = config['loss']['params']
        
        if loss_name == "mse":
            return F.mse_loss
        elif loss_name == "asymmetric":
            return lambda pred, true: asymmetric_loss(pred, true, alpha=loss_params['alpha'])
        elif loss_name == "huber":
            return lambda pred, true: F.huber_loss(pred, true, delta=loss_params['delta'])
        elif loss_name == "quantile":
            return lambda pred, true: quantile_loss(pred, true, beta=loss_params['beta'])
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

def asymmetric_loss(y_pred, y_true, alpha=1.0):
    """Penalizes over-predictions more than under-predictions"""
    diff = y_pred - y_true
    loss = torch.where(diff > 0, 
                      alpha * (diff ** 2),
                      diff ** 2)
    return loss.mean()

def quantile_loss(y_pred, y_true, beta=0.9):
    """Quantile regression loss for predicting specific quantiles"""
    diff = y_pred - y_true
    return torch.mean(torch.max(beta * diff, (beta - 1) * diff))