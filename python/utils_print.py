from etn_dataset import HierarchyDefinition
from enum import Enum
import torch
import numpy as np

__all__ = ['PlaybackMode', 'print_hierarchy', 'print_frame', 'SamplePrinter']


class PlaybackMode(Enum):
    SEQUENCE = 1
    FRAME = 2


def print_hierarchy(hierarchy: HierarchyDefinition, hierarchy_name: str, separator: str):
    out_string = f"H {hierarchy_name} "

    out_string += separator.join(hierarchy.bone_names)
    out_string += " "
    out_string += separator.join([str(p) for p in hierarchy.parent_ids])
    print(out_string)


def print_frame(root_pos, quats, rig_name: str, separator: str):
    """
    Prints a single frame of animation in the format "P <rig_name> <root_pos><separator><quats>".
    Note that message marker and rig name are separated with a whitespace, while other elements use the provided separator.

    :param root_pos: The global root position of the pose
    :param quats: The quaternion rotations of each joint in the pose
    :param rig_name: The name of the rig. (Can be used to run multiple animations at the same time)
    :param separator: The separator to use between message elements
    """

    frame_string = f"P {rig_name} "  # frame marker + rig identifier
    frame_string = frame_string + separator.join([str(p) for p in root_pos])  # root position
    frame_string += separator  # separator
    frame_string += separator.join([str(q) for q in quats])  # quats

    print(frame_string)


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


def process_sample_pair(org_quats: torch.Tensor,
                        pred_quats: torch.Tensor,
                        org_pos: torch.Tensor,
                        pred_pos: torch.Tensor,
                        org_contacts: torch.Tensor,
                        pred_contacts: torch.Tensor
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