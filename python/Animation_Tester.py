import pathlib
import numpy as np

from etn_dataset import ETNDataset
from utils_print import *


def run():
    # load dataset
    project_dir = str(pathlib.Path(__file__).parent.parent.absolute())
    data_path = f"{project_dir}/data/lafan1_reduced/train"
    data = ETNDataset(data_path, window_step=41)
    player = SampleExplorer(sample_size=32, anim_dataset=data)
    player.play()


class SampleExplorer:
    def __init__(self, sample_size, anim_dataset: ETNDataset):
        self.sample_size: int = sample_size
        self.data = anim_dataset
        self.mode = PlaybackMode.SEQUENCE
        self.running = True
        self.rig_name = "original"
        self.sample_idx = 0  # handles global sample index
        self.frame_idx = 0  # handles frame index in samples
        self.total_samples = len(self.data.animations)
        self.separator = ';'
        self.sample_length = 41  # The amount of frames in each sample. TODO: Base on database variable

    def play(self):

        print_hierarchy(self.data.hierarchy, self.rig_name, self.separator)

        while self.running:
            in_message = input("Cont?\n")
            message = in_message.split(self.separator)

            key = message[0]
            args = message[1:]

            mode = self.mode
            if key == "NAV":  # Navigate frames or sample indices
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
                elif args[0] == "FILE":  # Go to start of given file
                    self.sample_idx = self.data.get_index_of_filename(args[1])[0]

                # Wrap indices
                self.wrap_idx()

                sample = self.data.animations[self.sample_idx]

                self.print_sample(sample)

            elif key == "STOP":  # Stop playback and close program
                self.running = False

            elif key == "MODE":  # Change playback mode
                if args[0] == "FRAME":
                    self.mode = PlaybackMode.FRAME
                elif args[0] == "SEQ":
                    self.mode = PlaybackMode.SEQUENCE

            elif key == "FRAME":
                self.frame_idx = int(args[0])

            else:
                print("ERROR: could not parse message. Try again.\n")

    def wrap_idx(self):

        # Wrap frame_idx first since this can influence sample_idx
        if self.frame_idx > self.sample_length - 1:
            self.frame_idx = 0
            self.sample_idx += 1
        elif self.frame_idx < 0:
            self.frame_idx = self.sample_length - 1
            self.sample_idx -= 1

        if self.sample_idx > self.total_samples - 1:
            self.sample_idx = 0
        elif self.sample_idx < 0:
            self.sample_idx = self.total_samples - 1

    def print_sample(self, sample):
        """
        Prints an entire animation sample
        """

        root_vel = sample[0]
        quats = sample[1]
        root_offset = sample[2]
        quat_offset = sample[3]
        target_frame = sample[4]
        glob_positions = sample[5]
        contacts = sample[6]
        labels = sample[7]

        root_pos = glob_positions[:, :3]
        if self.mode == PlaybackMode.SEQUENCE:
            for i in range(0, len(quats)):
                print_frame(root_pos[i], quats[i], self.rig_name, self.separator)
                self.print_positions(glob_positions[i], self.rig_name)
                self.print_frame_debug(self.sample_idx, i, contacts[i], labels[i], self.rig_name, root_pos[i], root_vel[i])
                print("E")

        elif self.mode == PlaybackMode.FRAME:
            idx = self.frame_idx
            print_frame(root_pos[idx], quats[idx], self.rig_name, self.separator)
            self.print_positions(glob_positions[idx], self.rig_name)
            self.print_frame_debug(self.sample_idx, idx, contacts[idx], labels[idx], self.rig_name, root_pos[idx], root_vel[idx])

    def print_frame_debug(self, sample_idx, frame_idx, contacts, labels, rig_name: str, root_pos, root_vel):

        debug_string = f"A {rig_name} {frame_idx} "  # Marker must always be first element
        filename = self.data.get_filename_by_index(self.sample_idx)
        debug_string += f"Filename={filename}" + self.separator
        debug_string += f"Indices={self.data.get_index_of_filename(filename)[0]}-{self.data.get_index_of_filename(filename)[1]}" + self.separator
        debug_string += f"Sample-index={sample_idx}/{self.total_samples-1}" + self.separator
        debug_string += f"Frame-index={frame_idx}/{self.sample_length-1}" + self.separator
        debug_string += f"---------------" + self.separator
        debug_string += f"Contacts=[{','.join(['1' if x else '0' for x in contacts])}]" + self.separator
        vel_magn = np.sqrt(root_vel[0] ** 2 + root_vel[2] ** 2)
        debug_string += f"Root_vel={vel_magn}" + self.separator
        debug_string += f"Labels=[{self.separator + self.list_to_string(labels, self.separator)}]"

        print(debug_string)

    def list_to_string(self, list, sep:str) -> str:
        outstr = f"{sep.join(str(e) for e in list)}"
        return outstr

    def print_positions(self, glob_positions, rig_name:str):
        frame_string = f"G {rig_name} "  # frame marker + rig identifier
        frame_string += self.separator.join([str(j) for j in glob_positions])  # quats

        print(frame_string)

run()  # Encapsulate run behaviour to prevent globals
