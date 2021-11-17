import torch.nn
from torch import nn
import torch as t
from etn_dataset import HierarchyDefinition
from etn_encoderdecoder import *

from parametized_predictor import ParamPredictor
from parametized_lstm import ParamLSTM

class ETNGenerator(nn.Module):
    def __init__(self, hierarchy: HierarchyDefinition, device:str):
        super().__init__()

        self.device = device

        self.num_joints = hierarchy.bone_count()
        self.c_size = 4  # Contact information size (2 bools pr foot)
        self.q_size = self.num_joints * 4  # Size of quaternion pose vector
        self.r_size = 3  # Size of root-velocity vector
        self.o_size = self.num_joints * 4 + 3  # Size of offset vector

        # Define encoders
        self.state_encoder = Encoder(dims=(self.q_size + self.r_size + self.c_size, 512, 256))
        self.offset_encoder = Encoder(dims=(self.o_size, 512, 256))
        self.target_encoder = Encoder(dims=(self.q_size, 512, 256))
        self.encoded_size = 256 + 256 + 256  # Encoded vector is concatenated output of each encoder.

        # Define prediction layer (LSTM)
        self.predictor = ParamLSTM(input_size = self.encoded_size, hidden_size=512, device=self.device)

        # Define decoder
        self.decoder = Decoder(dims=(512, 512, 256, self.q_size + self.r_size + self.c_size))
        self.to(self.device)

    def forward(self,
                past_root_vel: t.Tensor,
                past_quats: t.Tensor,
                past_root_offset: t.Tensor,
                past_quat_offset: t.Tensor,
                past_contacts: t.Tensor,
                target_root_pos: t.Tensor,
                target_quats: t.Tensor,
                init_root_pos: t.Tensor,
                pred_weights,
                pred_bias
                ):

        lstm_state = None  # TODO: hidden state initializer here
        pred_poses = list()
        pred_contacts = list()
        glob_root = init_root_pos

        # Declaring vars to be assigned in for-loop to prevent compiler warnings
        next_rvel = None
        next_quats = None
        next_contacts = None
        next_root_offset = None
        next_quat_offsets = None
        lstm_states = None

        # Initialize with past-context
        for past_idx in range(10):
            next_rvel, next_quats, next_contacts, lstm_states = self.step(
                root_vel=past_root_vel[:, past_idx],
                quats=past_quats[:, past_idx],
                root_offset=past_root_offset[:, past_idx],
                quat_offset=past_quat_offset[:, past_idx],
                contacts=past_contacts[:, past_idx],
                target_quats=target_quats,
                prev_states = lstm_states,
                pred_weights=pred_weights,
                pred_bias=pred_bias
            )
            glob_root += past_root_vel[:, past_idx]  # NOTE: MIGHT BE ONE FRAME OFF
            next_root_offset = glob_root - target_root_pos
            next_quat_offsets = next_quats - target_quats

        # Final predicted frame of past-context is first frame of transition.
        first_frame = t.cat([next_rvel, next_quats], dim=1)
        pred_poses.append(first_frame)
        pred_contacts.append(next_contacts)

        # Predict transition poses
        for frame_idx in range(30 - 1):
            # Predict pose
            next_rvel, next_quats, next_contacts, lstm_states = self.step(
                root_vel=next_rvel,
                quats=next_quats,
                root_offset=next_root_offset,
                quat_offset=next_quat_offsets,
                contacts=next_contacts,
                target_quats=target_quats,
                prev_states=lstm_states,
                pred_weights=pred_weights,
                pred_bias=pred_bias
            )
            # Update offsets
            glob_root += next_rvel
            next_root_offset = glob_root - target_root_pos
            next_quat_offsets = next_quats - target_quats

            pred_poses.append(t.cat([glob_root, next_quats], dim=1))
            pred_contacts.append(next_contacts)

        # Generated transition is currently a list of tensors of (batch_size, frame_size) of len=transition_length,
        #   so must be stacked to be (batch_size, transition_length, frame_size)
        pred_poses = t.stack(pred_poses, dim=1)
        pred_contacts = t.stack(pred_contacts, dim=1)

        return pred_poses, pred_contacts

    def step(self,
             root_vel: t.Tensor,
             quats: t.Tensor,
             root_offset: t.Tensor,
             quat_offset: t.Tensor,
             target_quats: t.Tensor,
             contacts: t.Tensor,
             prev_states,
             pred_weights,
             pred_bias
             # TODO: tta input here
             ):
        """
        Performs a single step of the generator network, predicting a single pose. Note that a single forward
        pass of the generator is comprised of multiple steps.

        :param root_vel: The root_velocity
        :param quats: The current pose in local quaternions
        :param root_offset: The root-joint offset
        :param quat_offset: The pr-joint quaternion offset
        :param target_quats: The target pose in local quaternions
        :param contacts: The contact information vector.
        :param prev_state: prev LSTM state.
        :return: The root-velocity, local quaternions and contact info for the next pose. Also returns the LSTM state.
        """
        # Concatenate input vectors
        state_vec = t.cat([contacts, quats, root_vel], dim=1)
        offset_vec = t.cat([root_offset, quat_offset], dim=1)

        # Encode input vectors # TODO: z_tta and z_target
        h_state = self.state_encoder.forward(state_vec)
        h_offset = self.offset_encoder.forward(offset_vec)
        h_target = self.target_encoder.forward(target_quats)

        pred_in = t.cat([h_state, h_offset, h_target], dim=1)

        # Predict next pose
        pred_out, lstm_states = self.predictor.forward(pred_in, prev_states, pred_weights, pred_bias)

        # Decode prediction
        out = self.decoder.forward(pred_out)

        # Prediction format is a delta from prev pose. Add prev pose values to get next pose.
        next_quats = out[:, :self.q_size] + quats
        next_root_vel = out[:, self.q_size:self.q_size + self.r_size] + root_vel
        next_contacts = out[:, self.q_size + self.r_size:]  # Not a delta, bools.

        batch_size, num_elements = next_quats.shape
        next_quats = t.reshape(next_quats, (batch_size, num_elements // 4, 4))
        next_quats = t.nn.functional.normalize(next_quats, dim=2)
        next_quats = t.reshape(next_quats, (batch_size, num_elements))

        return next_root_vel, next_quats, t.sigmoid(next_contacts), lstm_states
