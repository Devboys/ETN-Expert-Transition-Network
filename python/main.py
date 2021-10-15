import sys
import pathlib
from pathlib import Path
import uuid
from torch.utils.data import DataLoader
import numpy as np
import torch as t

import utils_print
from etn_generator import ETNGenerator
from etn_dataset import ETNDataset
from utils_print import *


class SamplePrinter:
    def __init__(self, org_name, pred_name, separator, num_samples, sample_length):
        self.org_name = org_name
        self.pred_name = pred_name
        self.separator = separator
        self.sample_idx = 0
        self.frame_idx = 0
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.mode = PlaybackMode.SEQUENCE

    def start_print_loop(self, org_samples, pred_samples, org_labels):
        running = True
        while running:
            in_message = input("Cont?\n")
            message = in_message.split(self.separator)

            key = message[0]
            args = message[1:]

            mode = self.mode
            if key == "NAV":
                if args[0] == "NEXT":  # Increment index by 1
                    if mode == PlaybackMode.FRAME:
                        self.frame_idx += 1
                    elif mode == PlaybackMode.SEQUENCE:
                        self.sample_idx += 1
                elif args[0] == "PREV":  # Decrement index by 1
                    if mode == PlaybackMode.FRAME:
                        self.frame_idx -= 1
                    elif mode == PlaybackMode.SEQUENCE:
                        self.sample_idx -= 1
                elif args[0] == "GOTO":  # Go to given index
                    if mode == PlaybackMode.FRAME:
                        self.frame_idx = int(args[1])
                    elif mode == PlaybackMode.SEQUENCE:
                        self.sample_idx = int(args[1])

                self.wrap_idx()
                # self.print_sequence_pair(org_quats[self.sample_idx], pred_quats[self.sample_idx], org_pos[self.sample_idx], pred_pos[self.sample_idx], org_contacts[self.sample_idx], pred_contacts[self.sample_idx], org_labels[self.sample_idx])
                self.print_sequence_pair(org_samples, pred_samples, org_labels[self.sample_idx])

            elif key == "STOP":
                running = False

            elif key == "MODE":  # Change playback mode
                if args[0] == "FRAME":
                    self.mode = PlaybackMode.FRAME
                elif args[0] == "SEQ":
                    self.mode = PlaybackMode.SEQUENCE

            elif key == "FRAME":
                self.frame_idx = int(args[0])

    def wrap_idx(self):

        # Wrap frame_idx first since this can influence sample_idx
        if self.frame_idx > self.sample_length - 1:
            self.frame_idx = 0
            self.sample_idx += 1
        elif self.frame_idx < 0:
            self.frame_idx = self.sample_length - 1
            self.sample_idx -= 1

        if self.sample_idx > self.num_samples - 1:
            self.sample_idx = 0
        elif self.sample_idx < 0:
            self.sample_idx = self.num_samples - 1

    def print_sequence_pair(self, org_sample, pred_sample, label):

        org_quats = org_sample[0][self.sample_idx]
        org_pos = org_sample[1][self.sample_idx]
        org_contacts = org_sample[2][self.sample_idx]

        pred_quats = pred_sample[0][self.sample_idx]
        pred_pos = pred_sample[1][self.sample_idx]
        pred_contacts = pred_sample[2][self.sample_idx]

        if self.mode == PlaybackMode.SEQUENCE:
            for i in range(0, self.sample_length):
                print_frame(org_pos[i], org_quats[i], self.org_name, self.separator)
                self.print_frame_debug(org_contacts[i], self.org_name, label[i], i)
                print_frame(pred_pos[i], pred_quats[i], self.pred_name, self.separator)
                self.print_frame_debug(pred_contacts[i], self.pred_name, label[i], i)
                print("E")

        elif self.mode == PlaybackMode.FRAME:
            print_frame(org_pos[self.frame_idx], org_quats[self.frame_idx], self.org_name, self.separator)
            self.print_frame_debug(org_contacts[self.frame_idx], self.org_name, label[self.frame_idx], self.frame_idx)
            print_frame(pred_pos[self.frame_idx], pred_quats[self.frame_idx], self.pred_name, self.separator)
            self.print_frame_debug(pred_contacts[self.frame_idx], self.pred_name, label[self.frame_idx], self.frame_idx)

    def print_frame_debug(self, contacts, rig_name, label, frame_idx):
        debug_string = f"A {rig_name} {frame_idx} "  # Marker must always be first element
        debug_string += f"Sample-index={self.sample_idx}/{self.num_samples - 1}" + self.separator
        debug_string += f"Frame-index={frame_idx}/{self.sample_length - 1}" + self.separator
        debug_string += f"---------------" + self.separator
        debug_string += f"Contacts=[{','.join(['1' if x else '0' for x in contacts])}]" + self.separator
        debug_string += f"Labels=[{','.join(['1' if x else '0' for x in label])}]" + self.separator


        print(debug_string)


def run(base_dir, is_param_optimizing: bool):
    basedir = str(pathlib.Path(__file__).parent.parent.absolute())
    num_joints = 22
    model_id = str(uuid.uuid4())[:8] if is_param_optimizing else "NaN"
    tensorboard_dir = f"{base_dir}/tensorboard/"
    model_dir = f"{base_dir}/models/"

    # HYPERPARAMS
    batch_size = 32
    n_batches = 1  # i.e. training length
    learning_rate = 0.0005

    if is_param_optimizing:
        # for hyperparam optimization, learning rate is a random float between 0.0001 and 0.1
        power = -3 * np.random.rand()
        learning_rate = 10**(-1 + power)

    model_name = f"etn_{model_id}_bs{str(batch_size)}_nb{str(n_batches)}_lr{str(learning_rate)}.pt"
    model_path = model_dir + model_name

    train_data = ETNDataset(f"{basedir}/data/lafan1_reduced/train")
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = ETNDataset(f"{basedir}/data/lafan1_reduced/val", train_data=train_data)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    generator = ETNGenerator(
        hierarchy=val_data.hierarchy,
        learning_rate=learning_rate
    )

    if Path(model_path).exists():
        # Load model
        generator.load(model_path)
        print(f"Loaded model: {model_path}")

        # Get generated transition
        batch = next(iter(val_loader))
        pred_pos, pred_quats, pred_contacts = generator.eval_batch(batch)

        # Parse 'original' data aka batch. Used for comparison
        root, quats, root_offsets, quat_offsets, target_quats, ground_truth, global_positions, \
            contacts, labels = [b.float().to(generator.device) for b in batch]

        org_quats, pred_quats, org_pos, pred_pos, org_contacts, pred_contacts = process_sample_pair(quats,
                                                                                                    pred_quats,
                                                                                                    global_positions,
                                                                                                    pred_pos,
                                                                                                    contacts,
                                                                                                    pred_contacts)

        # Print hierarchy info
        separator = ';'
        org_name = "original"
        pred_name = "prediction"
        print_hierarchy(val_data.hierarchy, org_name, separator)
        print_hierarchy(val_data.hierarchy, pred_name, separator)

        org_samples = (org_quats, org_pos, org_contacts)
        pred_samples = (pred_quats, pred_pos, pred_contacts)

        sp = SamplePrinter(org_name, pred_name, separator, len(org_quats), len(org_quats[0]))
        sp.start_print_loop(org_samples, pred_samples, labels)
    else:
        generator.do_train(train_loader, model_id, tensorboard_dir, n_batches, val_loader)
        generator.save(model_path)


def process_sample_pair(org_quats: t.Tensor,
                        pred_quats: t.Tensor,
                        org_pos: t.Tensor,
                        pred_pos: t.Tensor,
                        org_contacts: t.Tensor,
                        pred_contacts: t.Tensor
                        ):
    """
    :return:
    """

    # Detach tensors
    org_quats = org_quats.detach().cpu().numpy()
    pred_quats = pred_quats.detach().cpu().numpy()
    org_pos = org_pos.detach().cpu().numpy()
    pred_pos = pred_pos.detach().cpu().numpy()
    org_contacts = org_contacts.detach().cpu().numpy()
    pred_contacts = pred_contacts.detach().cpu().numpy()

    # predicted quats include root_vel in [:3], so cut that out
    pred_quats = pred_quats[:, :, 3:]

    # Extract only root position from position vectors
    org_pos = org_pos[:, :, :3]
    pred_pos = pred_pos[:, :, :3]

    tar_frames = org_pos[:, [-1], :]

    # Append past context & target frame to predicted sequences
    pred_quats =    np.concatenate((org_quats[:, :10, :],    pred_quats[:, :, :],    org_quats[:, [-1], :]),    axis=1)
    pred_pos =      np.concatenate((org_pos[:, :10, :],      pred_pos[:, :, :],      org_pos[:, [-1], :]),      axis=1)
    pred_contacts = np.concatenate((org_contacts[:, :10, :], pred_contacts[:, :, :], org_contacts[:, [-1], :]), axis=1)

    return org_quats, pred_quats, org_pos, pred_pos, org_contacts, pred_contacts


run(sys.path[0], False)  # Encapsulate run behaviour to prevent globals

