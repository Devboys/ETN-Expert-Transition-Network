import torch as t
from torch import nn


class ETNGating(nn.Module):
    """
    Simple 2-hidden-layer MLP gating network. Produces blending coefficients for expert-blending.
    """
    def __init__(self, n_experts, device: str):
        super().__init__()

        r_size = 3  # Size of root-velocity vector
        l_size = 4  # Size of label-vector
        c_size = 4  # Size of contact-vector
        seq_length = 41  # TODO: global var

        input_size = (l_size + r_size + c_size) * seq_length
        h0_size = 512
        h1_size = 256
        output_size = n_experts

        # 4 layer NN with softmax output
        self.layers = nn.Sequential(
            nn.Linear(input_size, h0_size),
            nn.ELU(),
            nn.Linear(h0_size, h1_size),
            nn.ELU(),
            nn.Linear(h1_size, output_size),
            nn.Softmax(dim=1)
        )

        self.to(device)

    def forward(self, root_vel: t.Tensor, label: t.Tensor, contacts: t.Tensor):
        """
        The forward pass of the network. Takes as input the root_velocities, labels, and contacts for the entire
        sequence and produces blending coefficients for expert-blending
        :param root_vel: Root-velocity tensors for entire predicted sequence.
        :param label: Action-labels tensors for entire predicted sequence.
        :param contacts: Contact-tensors for entire predicted sequence
        :return: blending coefficient tensor of dims=[batch_size, n_experts]
        """

        b = root_vel.shape[0]  # num batches
        s = root_vel.shape[1]  # num frames in seq
        root_vel = t.reshape(root_vel, [b, s * 3])  # [b, s, 3] -> [b, s * 3]
        label = t.reshape(label, [b, s * 4])        # [b, s, 4] -> [b, s * 4]
        contacts = t.reshape(contacts, [b, s * 4])  # [b, s, 4] -> [b, s * 4]

        vec_in = t.cat([root_vel, label, contacts], dim=1)
        bc = self.layers.forward(vec_in)
        return bc
