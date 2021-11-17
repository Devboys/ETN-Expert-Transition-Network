import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as f


class ParamPredictor(nn.Module):
    # Very simple 2-hidden-layer functional MLP predictor that takes weights and biases as inputs.
    # Replaced by parametized_LSTM but kept here for posterity and/or simple reference

    def __init__(self, dims: tuple):
        super().__init__()
        self.dims = dims

    def forward(self, pred_input, weights, bias):
        x0 = torch.unsqueeze(pred_input, dim=2)

        x0 = t.matmul(weights[0], x0) + bias[0]
        x0 = f.leaky_relu(x0)

        x1 = t.matmul(weights[1], x0) + bias[1]
        x1 = f.leaky_relu(x1)

        x1 = torch.squeeze(x1, dim=2)

        return x1





