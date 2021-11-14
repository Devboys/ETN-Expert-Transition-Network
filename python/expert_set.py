import numpy as np
import torch as t


class ExpertSet:
    def __init__(self, layer_shape: tuple, device: str, rng: np.random.RandomState):
        # Shape expects (num_experts, in_size, out_size) of layer

        self.w_shape = layer_shape
        self.b_shape = (layer_shape[0], layer_shape[1], 1)

        self.weights = self.init_weights(rng)
        self.bias = self.init_bias()

        self.weights = t.nn.Parameter(self.weights.to(device))
        self.bias = t.nn.Parameter(self.bias.to(device))

    def init_weights(self, rng):
        rng_bound = np.sqrt(6 / np.prod(self.w_shape[1:]))
        weights = np.asarray(rng.uniform(low=-rng_bound, high=rng_bound, size=self.w_shape), dtype=np.float32)
        weights = t.tensor(weights, requires_grad=True)

        return weights

    def init_bias(self):
        return t.zeros(self.b_shape, requires_grad=True)

    def get_weight_blend(self, bc: t.Tensor, batch_size):
        w = t.unsqueeze(self.weights, 0)
        w = w.repeat([batch_size, 1, 1, 1])
        bc = t.unsqueeze(t.unsqueeze(bc, -1), -1)
        blend = bc * w
        return t.sum(blend, dim=1)

    def get_bias_blend(self, bc, batch_size):
        b = t.unsqueeze(self.bias, 0)
        b = b.repeat([batch_size, 1, 1, 1])
        bc = t.unsqueeze(t.unsqueeze(bc, -1), -1)
        blend = bc * b
        return t.sum(blend, dim=1)

    def get_parameters(self):
        return self.weights, self.bias
