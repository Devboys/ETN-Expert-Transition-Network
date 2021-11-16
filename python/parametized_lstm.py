import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as func


class ParamLSTM(nn.Module):
    # an LSTM implemented using only torch.functional. Weights (and hidden state) is expected to be supplied manually

    def __init__(self, input_size: int, hidden_size: int, device: str):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def get_dims(self):
        return self.input_size, self.hidden_size

    def forward(self, pred_input: t.Tensor, hidden_states,  weights, bias):
        x = torch.unsqueeze(pred_input, dim=2)

        if hidden_states is None:
            shape = (x.shape[0], self.hidden_size, 1)
            hidden_states = (t.zeros(shape).to(self.device), t.zeros(shape).to(self.device))

        h_state = hidden_states[0]
        c_state = hidden_states[1]

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





