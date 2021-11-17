import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as func


class ParamLSTM(nn.Module):
    """
    An LSTM implementation that has no internal state, but instead takes weights, biases, hidden and cell state as input.
    Implemented purely using torch.functional.
    """
    # an LSTM implemented using only torch.functional. Weights (and hidden state) is expected to be supplied manually

    def __init__(self, input_size: int, hidden_size: int, device: str):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def get_dims(self):
        """
        Returns a tuple of (input_size, hidden_size) for this lstm
        """
        return self.input_size, self.hidden_size

    def init_states(self, batch_size):
        """
        Initializes the hidden- & cell-state vectors as zero-tensors with dim=[batch_size, hidden_size, hidden_size]
        :return: A tuple of (hidden_state, cell_state)
        """
        shape = (batch_size, self.hidden_size, 1)
        h_state = t.zeros(shape).to(self.device)
        c_state = t.zeros(shape).to(self.device)
        return h_state, c_state


    def forward(self, pred_input: t.Tensor, lstm_states, weights, bias):
        """
        The forward pass of the defined LSTM. Predicts a single value.
        If lstm_states is null, they will be initialized.

        :param pred_input: Input vector
        :param lstm_states: Tuple of (hidden_state, cell_state). If null, will be initialized as zero-tensors.
        :param weights: List of LSTM weight-tensors with layout (w_hi, w_xi, w_hf, w_xf, w_hg, w_xg, w_ho, w_xo).
        :param bias: List of LSTM bias-tensors with layout (b_hi, b_xi, b_hf, b_xf, b_hg, b_xg, b_ho, b_xo).
        :return: A tuple of (prediction, lstm_states)
        """
        x = torch.unsqueeze(pred_input, dim=2)

        if lstm_states is None:
            lstm_states = self.init_states(x.shape[0])

        h_state = lstm_states[0]
        c_state = lstm_states[1]

        # LSTM parameters
        w_hi, w_xi = weights[0], weights[1]
        w_hf, w_xf = weights[2], weights[3]
        w_hg, w_xg = weights[4], weights[5]
        w_ho, w_xo = weights[6], weights[7]
        b_hi, b_xi = bias[0], bias[1]
        b_hf, b_xf = bias[2], bias[3]
        b_hg, b_xg = bias[4], bias[5]
        b_ho, b_xo = bias[6], bias[7]

        # Input Gate
        i_t = t.matmul(w_hi, h_state) + b_hi + t.matmul(w_xi, x) + b_xi
        i_t = t.sigmoid(i_t)

        # Forget Gate
        f_t = t.matmul(w_hf, h_state) + b_hf + t.matmul(w_xf, x) + b_xf
        f_t = t.sigmoid(f_t)

        # Cell Gate
        g_t = t.matmul(w_hg, h_state) + b_hg + t.matmul(w_xg, x) + b_xg
        g_t = t.tanh(g_t)

        # Output gate
        o_t = t.matmul(w_ho, h_state) + b_ho + t.matmul(w_xo, x) + b_xo
        o_t = t.sigmoid(o_t)

        # Update cell state
        c_state = t.mul(f_t, c_state) + t.mul(i_t, g_t)  # t.mul = element-wise

        # Get new hidden state
        h_state = t.mul(o_t, t.tanh(c_state))
        x_out = torch.squeeze(h_state, dim=2)

        return x_out, (h_state, c_state)





