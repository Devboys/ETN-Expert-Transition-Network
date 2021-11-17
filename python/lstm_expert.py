import numpy as np
import torch as t


class LSTMExpert:
    def __init__(self, input_size, hidden_size, num_experts, device: str, rng: np.random.RandomState):
        """
        Expert-param set intended for parametized_lstm implementation. \n
        Stores a number parameter-configurations and produces blends between them using get_weight_blend() and
        get_bias_blend().

        :param input_size: The input-layer size of the representative lstm-layer.
        :param hidden_size: The hidden-layer of the representative lstm-layer.
        :param num_experts: The number of expert weights to store.
        :param device: pytorch device-string for initialization.
        :param rng: A seeded numpy rng used for initializing parameters.
        """

        self.wx_shape = (num_experts, hidden_size, input_size)
        self.wh_shape = (num_experts, hidden_size, hidden_size)
        self.b_shape = (num_experts, hidden_size, 1)

        weights = list()
        bias = list()

        for n in range(0, 4):
            weight_hn = t.nn.Parameter(self.__init_weights(rng, self.wh_shape).to(device))
            weight_xn = t.nn.Parameter(self.__init_weights(rng, self.wx_shape).to(device))
            bias_hn = t.nn.Parameter(self.__init_bias(self.b_shape).to(device))
            bias_xn = t.nn.Parameter(self.__init_bias(self.b_shape).to(device))

            weights.append(weight_hn)
            weights.append(weight_xn)
            bias.append(bias_hn)
            bias.append(bias_xn)

        self.weights = weights
        self.bias = bias

        # EXPECTED:
        # w_hi, w_xi = weights[0], weights[1]
        # w_hf, w_xf = weights[2], weights[3]
        # w_hg, w_xg = weights[4], weights[5]
        # w_ho, w_xo = weights[6], weights[7]
        # b_hi, b_xi = bias[0], bias[1]
        # b_hf, b_xf = bias[2], bias[3]
        # b_hg, b_xg = bias[4], bias[5]
        # b_ho, b_xo = bias[6], bias[7]

    def __init_weights(self, rng, shape):
        rng_bound = np.sqrt(6 / np.prod(shape[1:]))
        weights = np.asarray(rng.uniform(low=-rng_bound, high=rng_bound, size=shape), dtype=np.float32)
        weights = t.tensor(weights, requires_grad=True)

        return weights

    def __init_bias(self, shape):
        return t.zeros(shape, requires_grad=True)

    def get_weight_blend(self, bc: t.Tensor, batch_size):
        """
        Blends the stored expert-weights using the provided blending-coefficients.
        :return: The blended weight configuration.
        """
        blend_list = list()
        for n in range(len(self.weights)):
            blend_n = self.__blend_single(self.weights[n], bc, batch_size)
            blend_list.append(blend_n)
        return blend_list

    def get_bias_blend(self, bc, batch_size):
        """
        Blends the stored expert-biases using the provided blending-coefficients.
        :return: The blended bias configuration.
        """
        blend_list = list()
        for n in range(len(self.weights)):
            blend_n = self.__blend_single(self.bias[n], bc, batch_size)
            blend_list.append(blend_n)
        return blend_list

    def __blend_single(self, param, bc, batch_size):
        x = t.unsqueeze(param, 0)
        x = x.repeat([batch_size, 1, 1, 1])
        bc = t.unsqueeze(t.unsqueeze(bc, -1), -1)
        blend = t.mul(bc, x)
        return t.sum(blend, dim=1)

    def get_parameters(self):
        return self.weights, self.bias

    def set_parameters(self, params):
        self.weights = params[0]
        self.bias = params[1]
