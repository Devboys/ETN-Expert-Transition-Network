import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

import utils_fk
import utils_tensorboard
from etn_gating import ETNGating
from etn_experts import ETNExperts
from etn_generator import ETNGenerator
from etn_dataset import HierarchyDefinition


class ETNModel(nn.Module):
    def __init__(self, name, hierarchy: HierarchyDefinition, learning_rate=1e-3):
        super().__init__()

        # Set values
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name

        self.bone_offsets = torch.from_numpy(np.tile(hierarchy.bone_offsets, (32, 1, 1))).float().to(
            self.device)  # Store separately as tensor
        self.parent_ids = torch.from_numpy(hierarchy.parent_ids).float().to(self.device)  # Store separately as tensor
        self.num_joints = hierarchy.bone_count()

        self.generator = ETNGenerator(hierarchy) # Temp location for adam

        self.epoch_idx = 0
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, amsgrad=True, betas=(0.5, 0.9))
        self.to(self.device)
        self.batch_idx = 0
        self.LOSS_MODIFIER_G = 0.1
        self.LOSS_MODIFIER_P = 0.5

        self.lr = learning_rate

        # Build Model
        # self.gating = ETNGating()
        # self.experts = ETNExperts()

    def do_train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, val_freq: int, tensorboard_dir: str):

        train_writer, val_writer = utils_tensorboard.get_writers(tensorboard_dir, self.name)
        train_iter = iter(train_loader)
        pbar = tqdm.tqdm(range(num_epochs))

        for _ in pbar:
            self.train()

            # Get batch data
            batch = next(train_iter)
            batch = [b.float().to(self.device) for b in batch]  # Convert to float values for concat

            root = batch[0]
            quats = batch[1]
            root_offsets = batch[2]
            quat_offsets = batch[3]
            target_quats = batch[4]
            ground_truth = batch[5]
            global_positions = batch[6]
            contacts = batch[7]
            labels = batch[8]

            # Predict sequence
            pred_poses, pred_contacts = self.forward(
                past_root=root[:, :10],
                past_quats=quats[:, :10],
                past_root_offset=root_offsets[:, :10],
                past_quat_offset=quat_offsets[:, :10],
                past_contacts=contacts[:, :10],
                target_quats=target_quats,
                target_root_pos=root_offsets[:, 0],
                init_root_pos=global_positions[:, 0, :3]
            )
            # To root-relative
            pred_poses_glob = self.fk(pred_poses)[:, :, 3:]
            rrrr = global_positions[:, 10:-1, 3:]


            # Calculate loss and backpropagate
            loss, l_quat, l_contact, l_pos = self.__loss(pred_poses, ground_truth[:, 10:-1], pred_contacts, contacts[:, 10:-1], pred_poses_glob,
                               rrrr)  # skipping last (target) frames
            assert torch.isfinite(loss), f"loss is not finite: {loss}"

            # Backpropagate and optimize
            loss.backward()  # Calculate gradients
            self.optimizer.step()  # Gradient descent
            self.optimizer.zero_grad()  # Reset stored gradient values
            train_writer.add_scalar("loss/loss", loss.item(), self.epoch_idx)  # Log loss
            pbar.set_description(f"Training generator. Loss {loss}, l_pos={l_pos}, l_quat={l_quat}, l_contact={l_contact}")  # Update progress bar

            # Validate
            if self.epoch_idx % val_freq == 0:
                self.do_validation(val_loader, val_writer)
            self.epoch_idx += 1

    def forward(self,
                past_root: torch.Tensor,
                past_quats: torch.Tensor,
                past_root_offset: torch.Tensor,
                past_quat_offset: torch.Tensor,
                past_contacts: torch.Tensor,
                target_root_pos: torch.Tensor,
                target_quats: torch.Tensor,
                init_root_pos: torch.Tensor
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

        # Initialize with past-context
        for past_idx in range(10):
            next_rvel, next_quats, next_contacts, lstm_state = self.__pred_pose(
                root_vel=past_root[:, past_idx],
                quats=past_quats[:, past_idx],
                root_offset=past_root_offset[:, past_idx],
                quat_offset=past_quat_offset[:, past_idx],
                contacts=past_contacts[:, past_idx],
                target_quats=target_quats,
                prev_state=lstm_state
            )
            if glob_root is None:
                glob_root = next_rvel
            else:
                glob_root += next_rvel
            next_root_offset = glob_root - target_root_pos
            next_quat_offsets = next_quats - target_quats

        # Final predicted frame of past-context is first frame of transition.
        first_frame = torch.cat([next_rvel, next_quats], dim=1)
        pred_poses.append(first_frame)
        pred_contacts.append(next_contacts)

        # Predict transition poses
        for frame_idx in range(30 - 1):
            # Predict pose
            next_rvel, next_quats, next_contacts, lstm_state = self.__pred_pose(
                root_vel=next_rvel,
                quats=next_quats,
                root_offset=next_root_offset,
                quat_offset=next_quat_offsets,
                contacts=next_contacts,
                target_quats=target_quats,
                prev_state=lstm_state
            )
            # Update offsets
            glob_root += next_rvel
            next_root_offset = glob_root - target_root_pos
            next_quat_offsets = next_quats - target_quats

            pred_poses.append(torch.cat([next_rvel, next_quats], dim=1))
            pred_contacts.append(next_contacts)

        # Generated transition is currently a list of tensors of (batch_size, frame_size) of len=transition_length,
        #   so must be stacked to be (batch_size, transition_length, frame_size)
        pred_poses = torch.stack(pred_poses, dim=1)
        pred_contacts = torch.stack(pred_contacts, dim=1)

        return pred_poses, pred_contacts

    def __pred_pose(self, root_vel, quats, root_offset, quat_offset, contacts, target_quats, prev_state):
        # Get blending coefficients from gating network
        # TODO
        # Get NN weights from experts
        # TODO
        # Set generator weights and predict next frame
        # self.generator.set_weights()

        prediction = self.generator.step(
            root_vel=root_vel,
            quats=quats,
            root_offset=root_offset,
            quat_offset=quat_offset,
            quat_target=target_quats,
            contacts=contacts,
            prev_state=prev_state
        )

        # Return pred pose tuple
        return prediction

    def do_validation(self, loader: DataLoader, writer: SummaryWriter):

        self.eval()  # Set network to eval mode.
        batch = next(iter(loader))
        batch = [b.float().to(self.device) for b in batch]  # Convert to float values for concat

        root = batch[0]
        quats = batch[1]
        root_offsets = batch[2]
        quat_offsets = batch[3]
        target_quats = batch[4]
        ground_truth = batch[5]
        global_positions = batch[6]
        contacts = batch[7]
        labels = batch[8]

        with torch.no_grad():  # Make no gradient calculations
            # Predict sequence
            pred_poses, pred_contacts = self.forward(
                past_root=root[:, :10],
                past_quats=quats[:, :10],
                past_root_offset=root_offsets[:, :10],
                past_quat_offset=quat_offsets[:, :10],
                past_contacts=contacts[:, :10],
                target_quats=target_quats,
                target_root_pos=-root_offsets[:, 0],
                init_root_pos=global_positions[:, 0, :3]
            )
            pred_positions = self.fk(pred_poses)

            # Calculate loss
            loss, l_quat, l_contact, l_pos = self.__loss(
                frames=pred_poses,
                gt_frames=ground_truth[:, 10:-1],  # skip last (target) frame
                contacts=pred_contacts,
                gt_contacts=contacts[:, 10:-1],  # skip last (target) frame
                positions=pred_positions,
                gt_positions=global_positions[:, 10:-1]  # skip last (target) frame
            )

            # Report loss
            writer.add_scalar("loss/loss", loss.item(), self.epoch_idx)

    def fk(self, frames):
        """
        Performs FK calculations on the given pose using the stored generators hierarchy.

        :param frames: Joint quats for pose
        :return: Calculated global joint positions
        """
        batch_size, frame_count, channels = frames.shape
        # Merge dimensions 0 & 1 to make one batch
        frames = frames.view(-1, channels)
        # Do FK
        frames = utils_fk.torch_forward_kinematics_batch(
            offsets=torch.cat(frame_count * [self.bone_offsets]),
            pose=frames,
            parents=self.parent_ids,  # TODO: removed [0] verify that its ok
            joint_count=self.num_joints
        )

        # Reshape back to original dimensions
        frames = frames.view(batch_size, frame_count, self.num_joints*3)
        return frames

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
        mae = lambda x, y: torch.mean(torch.abs(y - x))  # Mean absolute error (MAE)
        l_quat = mae(frames, gt_frames)
        l_contact = mae(contacts, gt_contacts) * self.LOSS_MODIFIER_G
        l_pos = mae(positions, gt_positions) * self.LOSS_MODIFIER_P

        loss = l_quat + l_contact + l_pos
        return loss, l_quat, l_contact, l_pos

    def save(self, filename):
        """
        Save the model to a file
        :param filename: name out output file
        """
        torch.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """
        Load a saved model

        :param filename: The name of the model to load
        """
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batch_idx = checkpoint["batch_idx"]


