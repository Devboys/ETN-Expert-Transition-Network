import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch as t
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils_fk import torch_forward_kinematics_batch
from utils_tensorboard import *
from etn_dataset import HierarchyDefinition


class Encoder(nn.Module):
    def __init__(self, input_size, h0_size, output_size):
        """
        Creates a simple 3-layer encoder network.

        :param input_size: num nodes in input layer
        :param h0_size: num nodes in hidden layer
        :param output_size: num nodes in output layer
        """
        super().__init__()
        self.input_size = input_size
        self.h0_size = h0_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, h0_size),
            nn.LeakyReLU(),
            nn.Linear(h0_size, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, input_vec: t.Tensor) -> t.Tensor:
        """
        The forward operation of the encoder network.

        :param input_vec: The input vector
        :return: An encoded input vector.
        """
        return self.layers(input_vec)


class Decoder(nn.Module):
    def __init__(self, input_size, h0_size, h1_size, output_size):
        """
        Creates a simple 4-layer decoder network.
        :param input_size: Num nodes in input layer
        :param h0_size: num nodes in first hidden layer
        :param h1_size: num nodes in second hidden layer
        :param output_size: num nodes in output layer
        """
        super().__init__()
        self.input_size = input_size
        self.h0_size = h0_size
        self.h1_size = h1_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, self.h0_size),
            nn.LeakyReLU(),
            nn.Linear(self.h0_size, self.h1_size),
            nn.LeakyReLU(),
            nn.Linear(self.h1_size, self.output_size)
        )

    def forward(self, input_vec: t.Tensor) -> t.Tensor:
        """
        The forward operation of the encoder network.

        :param input_vec: The input vector
        :return: An encoded input vector.
        """
        return self.layers(input_vec)


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

        # TODO (2): gan here
        self.optimizer = t.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True, betas=(0.5, 0.9))
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.to(self.device)
        self.batch_idx = 0
        self.LOSS_MODIFIER_G = 0.1
        self.LOSS_MODIFIER_P = 0.5

        self.train_writer = None
        self.val_writer = None

        self.lr = learning_rate

        self.bone_offsets = t.from_numpy(np.tile(hierarchy.bone_offsets, (32, 1, 1))).float().to(self.device)  # Store separately as tensor
        self.parent_ids = t.from_numpy(hierarchy.parent_ids).float().to(self.device)  # Store separately as tensor

    def __step(self,
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

    def __forward(self,
                  root_in: t.Tensor,
                  quats_in: t.Tensor,
                  root_offset_in: t.Tensor,
                  quat_offsets_in: t.Tensor,
                  target_quats: t.Tensor,
                  contacts_in: t.Tensor,
                  target_root_pos: t.Tensor,
                  transition_length=30
                  ):
        """
        Performs an entire forward pass of the network, stepping through the past context and predicting an entire
        animation transition.

        :param root_in: Past context root velocities
        :param quats_in: Past context joint rotations
        :param root_offset_in: Past context root offset
        :param quat_offsets_in: Past context joint-rotation offsets
        :param target_quats: Target pose in local quaternions
        :param contacts_in: Past context foot contacts
        :param target_root_pos: Target pose root global pos
        :return: Root pos, quaternion poses and contact info for the predicted sequence (only predicted poses, not past
            context)
        """

        lstm_state = None  # TODO: hidden state initializer here
        output_frames = list()
        sequence_contacts = list()
        glob_root = None

        # Declaring vars to be assigned in for-loop to prevent compiler warnings
        root = None
        quats = None
        contacts = None
        root_offset = None
        quat_offsets = None
        # TODO: generate base gauss noise for z_target calculations in step-function here.

        # PAST CONTEXT
        for frame_index in range(quats_in.shape[1]):
            # tta = 40 - frame_index

            # Step through past context frames to initialize vars for first frame of transition
            root, quats, contacts, lstm_state = self.__step(
                root_vel=root_in[:, frame_index],
                quats=quats_in[:, frame_index],
                root_offset=root_offset_in[:, frame_index],
                quat_offset=quat_offsets_in[:, frame_index],
                quat_target=target_quats,
                contacts=contacts_in[:, frame_index],
                prev_state=lstm_state
                # TODO: tta input here
            )
            # Update offsets
            if glob_root is None:
                glob_root = root
            else:
                glob_root += root
            root_offset = glob_root - target_root_pos
            quat_offsets = quats - target_quats

        # Save last output since that is the first frame of our transition
        first_frame = t.cat([root, quats], dim=1)
        output_frames.append(first_frame)
        sequence_contacts.append(contacts)

        # TRANSITION
        for frame_index in range(transition_length - 1):  # -1 because last frame of past-context predicts the first frame of transition.
            # tta = 30 - frame_index
            root, quats, contacts, lstm_state = self.__step(
                root_vel=root,
                quats=quats,
                root_offset=root_offset,
                quat_offset=quat_offsets,
                quat_target=target_quats,
                contacts=contacts,
                prev_state=lstm_state
                # TODO: tta input here
            )
            # Update offsets
            glob_root += root
            root_offset = glob_root - target_root_pos
            quat_offsets = quats - target_quats

            output_frames.append(t.cat([root, quats], dim=1))
            sequence_contacts.append(contacts)

        # Generated transition is currently a list of tensors of (batch_size, frame_size) of len=transition_length,
        #   so must be stacked to be (batch_size, transition_length, frame_size)
        output_frames = t.stack(output_frames, dim=1)
        sequence_contacts = t.stack(sequence_contacts, dim=1)

        return output_frames, sequence_contacts

    def __loss(self, frames, gt_frames, contacts, gt_contacts, positions, gt_positions):
        """
        Calculates loss between predicted transition and ground-truth (gt) frames.

        :param frames: Predicted transition frames (root vel + local quats)
        :param gt_frames: Ground truth frames
        :param contacts: Predicted contact info
        :param gt_contacts: Ground truth contact info
        :param positions: Predicted global joint positions
        :param gt_positions: Ground truth global joint positions
        :return: Accumulated loss of the entire transition
        """
        mae = lambda x, y: t.mean(t.abs(y - x))  # Mean absolute error (MAE)

        loss = mae(frames, gt_frames)
        loss += mae(contacts, gt_contacts) * self.LOSS_MODIFIER_G
        loss += mae(positions, gt_positions) * self.LOSS_MODIFIER_P
        return loss

    def eval_batch(self, batch):
        """
        Perform a network forward pass on the provided batch in eval mode, generating a single animation transition.

        :param batch: A concatenated input batch from an ETNDataset.
        :return: The predicted transition frames (excluding past context) for the given batch as well as used
            parent-hierarchy
            (Global joint positions, local joint quats)
        """
        self.eval()  # Flag network for eval mode
        root, quats, root_offsets, quat_offsets, target_quats, ground_truth, global_positions, \
            contacts, labels = [b.float().to(self.device) for b in batch]
        with t.no_grad():
            pred_quats, out_contacts = self.__forward(
                root_in=root[:, :10],
                quats_in=quats[:, :10],
                root_offset_in=root_offsets[:, :10],
                quat_offsets_in=quat_offsets[:, :10],
                target_quats=target_quats,
                contacts_in=contacts[:, :10],
                target_root_pos=-root_offsets[:, 0]
            )

            pred_positions = self.__fk(pred_quats)

            loss = self.__loss(
                frames=pred_quats,
                gt_frames=ground_truth[:, 10:-1],  # skip last (target) frame
                contacts=out_contacts,
                gt_contacts=contacts[:, 10:-1],  # skip last (target) frame
                positions=pred_positions,
                gt_positions=global_positions[:, 10:-1]  # skip last (target) frame
            )
        self.__report_loss(self.val_writer, loss)

        return pred_positions, pred_quats, out_contacts

    def __fk(self, frames):
        """
        Performs FK calculations on the given pose using the stored generators hierarchy.

        :param frames: Joint quats for pose
        :return: Calculated global joint positions
        """
        batch_size, frame_count, channels = frames.shape
        # Merge dimensions 0 & 1 to make one batch
        frames = frames.view(-1, channels)
        # Do FK
        offset_thing = self.bone_offsets

        frames = torch_forward_kinematics_batch(
            offsets=t.cat(frame_count*[offset_thing]),
            pose=frames,
            parents=self.parent_ids,  # TODO: removed [0] verify that its ok
            joint_count=self.num_joints
        )

        # Reshape back to original dimensions
        frames = frames.view(batch_size, frame_count, self.num_joints*3)
        return frames

    def __report_loss(self, writer: SummaryWriter, loss):
        """
        Writes a loss scalar to the given summary writer.
        :param writer: The SummaryWriter to write to
        :param loss: The loss to report.
        """
        if writer is not None:
            writer.add_scalar("loss/loss", loss.item(), self.batch_idx)

    def do_train(self,
                 train_loader: DataLoader,
                 model_id: str,
                 log_dir: str,
                 n_train_batches: int,
                 val_loader: DataLoader,
                 val_frequency=10,
                 ):

        # Set train and val writer
        self.train_writer, self.val_writer = get_writers(log_dir, f"etn_{model_id}_lr{self.lr}")

        data_iter = iter(train_loader)
        pbar = tqdm.tqdm(range(n_train_batches))

        for _ in pbar:
            # Switch to training mode
            self.train()

            # Extract batch info
            root, quats, root_offsets, quat_offsets, target_quats, ground_truth, \
                global_positions, contacts, labels = [b.float().to(self.device) for b in next(data_iter)]

            # Generate sequence
            poses, c = self.__forward(
                root_in=root[:, :10],
                quats_in=quats[:, :10],
                root_offset_in=root_offsets[:, :10],
                quat_offsets_in=quat_offsets[:, :10],
                target_quats=target_quats,
                contacts_in=contacts[:, :10],
                target_root_pos=-root_offsets[:, 0]
            )
            glob_poses = self.__fk(poses)

            # Calculate loss
            loss = self.__loss(poses, ground_truth[:, 10:-1], c, contacts[:, 10:-1], glob_poses, global_positions[:, 10:-1])  # skipping last (target) frames
            assert t.isfinite(loss), f"loss is not finite: {loss}"

            # Backpropagate and optimize
            loss.backward()
            self.optimizer.step()
            self.__report_loss(self.train_writer, loss)
            pbar.set_description(f"Training generator. Loss {loss}")
            self.optimizer.zero_grad()

            # TODO (2): gan here

            # Run validation at freq
            if self.batch_idx % val_frequency == 0:
                batch = next(iter(val_loader))
                self.eval_batch(batch)
            self.batch_idx += 1

    def save(self, filename):
        """
        Save the model to a file
        :param filename: name out output file
        """
        t.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """
        Load a saved model

        :param filename: The name of the model to load
        """
        checkpoint = t.load(filename, map_location=t.device(self.device))
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batch_idx = checkpoint["batch_idx"]


