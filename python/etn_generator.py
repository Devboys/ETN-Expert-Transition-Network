from torch import nn
import torch as t
from etn_dataset import HierarchyDefinition
from etn_encoderdecoder import *


class ETNGenerator(nn.Module):
    def __init__(self, hierarchy: HierarchyDefinition, learning_rate=1e-3):
        """

        :param hierarchy: The HierarchyDefinition of the data to work with.
        :param learning_rate:
        """
        super().__init__()

        self.num_joints = hierarchy.bone_count()
        self.c_size = 4  # Contact information size (2 bools pr foot)
        self.q_size = self.num_joints * 4  # Size of quaternion pose vector
        self.r_size = 3  # Size of root-velocity vector
        self.o_size = self.num_joints * 4 + 3  # Size of offset vector

        # Define encoders
        self.state_encoder = Encoder(self.q_size + self.r_size + self.c_size, 512, 256)
        self.offset_encoder = Encoder(self.o_size, 512, 256)
        self.target_encoder = Encoder(self.q_size, 512, 256)
        self.encoded_size = 256 + 256 + 256  # Encoded vector is concatenated output of each encoder.

        # Define prediction layer (LSTM)
        self.lstm = nn.LSTMCell(input_size=self.encoded_size, hidden_size=512)

        # Define decoder
        self.decoder = Decoder(512, 512, 256, self.q_size + self.r_size + self.c_size)

    def step(self,
             root_vel: t.Tensor,
             quats: t.Tensor,
             root_offset: t.Tensor,
             quat_offset: t.Tensor,
             quat_target: t.Tensor,
             contacts: t.Tensor,
             prev_state: (t.Tensor, t.Tensor)
             # TODO: tta input here
             ):
        """
        Performs a single step of the generator network, predicting a single pose. Note that a single forward
        pass of the generator is comprised of multiple steps.

        :param root_vel: The root_velocity
        :param quats: The current pose in local quaternions
        :param root_offset: The root-joint offset
        :param quat_offset: The pr-joint quaternion offset
        :param quat_target: The target pose in local quaternions
        :param contacts: The contact information vector.
        :param prev_state: prev LSTM state.
        :return: The root-velocity, local quaternions and contact info for the next pose. Also returns the LSTM state.
        """
        # TODO: Embeddings here (z_tta and z_target)
        # Concatenate input vectors
        state_vec = t.cat([contacts, quats, root_vel], dim=1)
        offset_vec = t.cat([root_offset, quat_offset], dim=1)

        # Encode input vectors
        h_state = self.state_encoder.forward(state_vec)  # TODO: add z_tta
        h_offset = self.offset_encoder.forward(offset_vec)  # TODO: add z_tta
        h_target = self.target_encoder.forward(quat_target)  # TODO: add z_tta
        h_offset_target = t.cat([h_offset, h_target], dim=1)  # TODO: add z_target here

        # Predict next pose
        lstm_input = t.cat([h_state, h_offset_target], dim=1)
        h_lstm, cell_state = self.lstm(lstm_input, None if prev_state is None else prev_state)

        # Decode prediction
        out = self.decoder.forward(h_lstm)

        # Prediction format is a delta from prev pose. Add prev pose values to get next pose.
        next_root_vel = out[:, self.q_size:self.q_size + self.r_size] + root_vel
        next_contacts = out[:, self.q_size + self.r_size:]  # Not a delta, bools.
        next_quats = out[:, :self.q_size] + quats

        batch_size, num_elements = next_quats.shape
        next_quats = t.reshape(next_quats, (batch_size, num_elements // 4, 4))
        next_quats = t.nn.functional.normalize(next_quats, dim=2)
        next_quats = t.reshape(next_quats, (batch_size, num_elements))

        return next_root_vel, next_quats, t.sigmoid(next_contacts), (h_lstm, cell_state)

    def set_weights(self):
        raise NotImplementedError
