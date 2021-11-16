import numpy as np
import numpy.random as rand
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

import utils_fk
import utils_tensorboard
from etn_gating import ETNGating
from lstm_expert import LSTM_Expert
from etn_generator import ETNGenerator
from etn_dataset import HierarchyDefinition


class ETNModel(nn.Module):
    def __init__(self, name, hierarchy: HierarchyDefinition, batch_size: int, rng: rand.RandomState, learning_rate=1e-3, n_experts: int = 4):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.batch_size = batch_size

        # Loss Vars
        self.LOSS_MODIFIER_Q = 1
        self.LOSS_MODIFIER_G = 0.1
        self.LOSS_MODIFIER_P = 0.5

        # FK Vars
        self.bone_offsets = torch.from_numpy(np.tile(hierarchy.bone_offsets, (batch_size, 1, 1))).float().to(
            self.device)  # Store separately as tensor
        self.parent_ids = torch.from_numpy(hierarchy.parent_ids).float().to(self.device)  # Store separately as tensor
        self.num_joints = hierarchy.bone_count()

        # Training vars
        self.batch_idx = 0
        self.epoch_idx = 0

        # Build Model
        self.gating = ETNGating(n_experts, self.device)
        self.generator = ETNGenerator(hierarchy, self.device)

        # Generator expert weights
        gen_dims = self.generator.predictor.get_dims()  # expert weights must match generator dimensions.
        self.lstm_expert = LSTM_Expert(gen_dims[0], gen_dims[1], n_experts, self.device, rng)

        params = list(self.parameters()) + list(self.lstm_expert.get_parameters()[0]) + list(self.lstm_expert.get_parameters()[1])  # include experts parameters
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True, betas=(0.5, 0.9))

        self.to(self.device)

    def do_train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, val_freq: int, tensorboard_dir: str):

        train_writer, val_writer = utils_tensorboard.get_writers(tensorboard_dir, self.name)
        train_iter = iter(train_loader)
        pbar = tqdm.tqdm(range(num_epochs))

        for _ in pbar:
            self.train()

            # Get batch data
            batch = next(train_iter)
            batch = [b.float().to(self.device) for b in batch]  # Convert to float values for concat
            root_vel     = batch[0]
            quats        = batch[1]
            root_offsets = batch[2]
            quat_offsets = batch[3]
            target_quats = batch[4]
            ground_truth = batch[5]
            glob_pos     = batch[6]
            contacts     = batch[7]
            labels       = batch[8]

            # Get blending coefficients through gating network

            # Predict sequence using expert blend
            pred_poses, pred_contacts = self.forward(root_vel, quats, root_offsets, quat_offsets, target_quats, glob_pos, contacts, labels)

            fk_pred_poses = pred_poses
            pred_poses_glob = self.fk(fk_pred_poses)

            # Calculate loss and backpropagate
            loss, l_quat, l_contact, l_pos = self.__loss(pred_poses[:, :, 3:], ground_truth[:, 10:-1, 3:], pred_contacts, contacts[:, 10:-1], pred_poses_glob,
                               glob_pos[:, 10:-1])  # skipping last (target) frames
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

    def do_validation(self, loader: DataLoader, writer: SummaryWriter):

        self.eval()  # Set network to eval mode.
        batch = next(iter(loader))
        batch = [b.float().to(self.device) for b in batch]  # Convert to float values for concat

        root_vel = batch[0]
        quats = batch[1]
        root_offsets = batch[2]
        quat_offsets = batch[3]
        target_quats = batch[4]
        ground_truth = batch[5]
        global_positions = batch[6]
        contacts = batch[7]
        labels = batch[8]

        with torch.no_grad():   # Make no gradient calculations

            # Get blending coefficients through gating network
            pred_poses, pred_contacts = self.forward(root_vel, quats, root_offsets, quat_offsets, target_quats, global_positions, contacts, labels)
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

    def forward(self, root_vel, quats, root_offsets, quat_offsets, target_quats, glob_pos, contacts, labels):
        # Get blending coefficients through gating network
        bc = self.gating.forward(root_vel, labels, contacts)

        # Get blended parameters from experts
        pred_weights = self.lstm_expert.get_weight_blend(bc, self.batch_size)
        pred_bias = self.lstm_expert.get_bias_blend(bc, self.batch_size)

        # Predict sequence using expert blend
        pred_poses, pred_contacts = self.generator.forward(
            past_root_vel=root_vel[:, :10],
            past_quats=quats[:, :10],
            past_root_offset=root_offsets[:, :10],
            past_quat_offset=quat_offsets[:, :10],
            past_contacts=contacts[:, :10],
            target_quats=target_quats,
            target_root_pos=root_offsets[:, 0],
            init_root_pos=glob_pos[:, 0, :3],
            pred_weights=pred_weights,
            pred_bias=pred_bias
        )

        return pred_poses, pred_contacts

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
            parents=self.parent_ids,
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
        l_quat = mae(frames, gt_frames) * self.LOSS_MODIFIER_Q
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
            'expert_gen_l0': self.gen_l0.get_parameters(),
            'expert_gen_l1': self.gen_l1.get_parameters()
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
        self.gen_l0.set_parameters(checkpoint['expert_gen_l0'])
        self.gen_l1.set_parameters(checkpoint['expert_gen_l1'])



