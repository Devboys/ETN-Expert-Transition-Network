import collections as c
import torch as t


class ExpertSet:
    def __init__(self, layerShape: tuple, device: str):
        # Shape expects (num_experts, in_size, out_size) of layer

        self.w_shape = layerShape
        self.b_shape = (layerShape[0], layerShape[1], 1)

        self.weights = self.init_weights()
        self.bias = self.init_bias()

        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)

    def init_weights(self):
        # TODO: seeded rng for reproducability
        return t.rand(self.w_shape)

    def init_bias(self):
        return t.zeros(self.b_shape)

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

    #OLD IMPLEMENT
    # def get_blended_params(self, bc: t.Tensor):
    #     # MAYBE I HAVE TO ACHIEVE USING TORCH METHODS FOR BACKPROP TO REGISTER. Custom Backward()?
    #     # BELOW WILL NOT BACKPROP, load_state_dict breaks computation graph. this would have to provide a parameter() list instead.
    #
    #     blended_dicts = [c.OrderedDict()] * bc.shape[0] # one dict for each bc in the batch
    #
    #     for key in self.experts[0]:
    #         v = self.experts[0][key]
    #         b = bc[:, 0:1]
    #         val = self.__blend_mat(v, b)
    #
    #         # blended element m = sum^0_n(e_n[m] * bc[n]), n = expert index
    #         for idx in range(1, self.n_experts):
    #             v = self.experts[idx][key]
    #             b = bc[:, idx:idx+1]
    #             e_val = self.__blend_mat(v, b)
    #             val += e_val
    #
    #         for idx2 in range(len(blended_dicts)):
    #             blended_dicts[idx][key] = val[idx]
    #
    #     return blended_dicts
    #
    # def __blend_mat(self, v, b):
    #     if(len(v.shape) > 1):  # bias list is 1D
    #         b = b.unsqueeze(1)
    #     shape = list(v.shape)
    #     shape.insert(0, 1)
    #     b = b.repeat(shape)
    #     val = t.mul(v, b)
    #     return val
