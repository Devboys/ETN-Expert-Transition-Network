import pathlib
import numpy as np

from etn_dataset import ETNDataset
from etn_dataset import HierarchyDefinition
from enum import Enum

# Take animation, play it in unity in a simple manner.
# Should be able to go back and forth
# Info to show:
# - Animation data (joint positions & rotation)
# - Current anim and/or frame index in dataset
# - Additional info packages (action label matrix, etc)


def run():
    # load dataset
    project_dir = str(pathlib.Path(__file__).parent.parent.absolute())
    data_path = f"{project_dir}/data/lafan1_reduced/train"
    data = ETNDataset(data_path)
    player = SampleExplorer(sample_size=32, anim_dataset=data)
    player.play()


class PlaybackMode(Enum):
    SEQUENCE = 1
    FRAME = 2


class SampleExplorer:
    def __init__(self, sample_size, anim_dataset: ETNDataset):
        self.sample_size: int = sample_size
        self.data = anim_dataset
        self.mode = PlaybackMode.FRAME
        self.running = True
        self.rig_name = "original"
        self.sample_idx = 0  # handles global sample index
        self.frame_idx = 0  # handles frame index in samples
        self.total_samples = len(self.data.animations)
        self.separator = ';'
        self.sample_length = 41  # The amount of frames in each sample. TODO: Base on database variable

    def play(self):

        self.print_hierarchy(self.data.hierarchy, self.rig_name)

        while self.running:
            in_message = input("Cont?\n")
            message = in_message.split(self.separator)

            try:
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

                    sequence = self.data.animations[self.sample_idx]
                    self.print_sample(sequence)

                elif key == "STOP":  # Stop playback and close program
                    self.running = False

                elif key == "MODE":  # Change playback mode
                    if args[0] == "FRAME":
                        self.mode = PlaybackMode.FRAME
                    elif args[0] == "SEQ":
                        self.mode = PlaybackMode.SEQUENCE

                else:
                    print("ERROR: could not parse message. Try again.\n")

            except Exception as e:
                print(f"Caught error with input: {in_message}")
                print(f"Error Message:", e)

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
                self.print_frame(root_pos[i], quats[i], self.rig_name)
                self.print_frame_debug(self.sample_idx, i, contacts[i], labels[i])

        elif self.mode == PlaybackMode.FRAME:
            idx = self.frame_idx
            self.print_frame(root_pos[idx], quats[idx], self.rig_name)
            self.print_frame_debug(self.sample_idx, idx, contacts[idx], labels[idx])

    def print_frame(self, root_pos, quats, rig_name: str):
        """
        Prints a single frame of animation in the format "P <rig_name> <root_pos>-<quats>".
        Each element of root_pos and every joint rotation in <quats> is separated with a '-'.

        :param root_pos: The global root position of the pose
        :param quats: The quaternion rotations of each joint in the pose
        :param rig_name: The name of the rig. (Can be used to run multiple animations at the same time)
        """

        frame_string = f"P {rig_name} "  # frame marker + rig identifier
        frame_string = frame_string + self.separator.join([str(p) for p in root_pos])  # root position
        frame_string += self.separator  # separator
        frame_string += self.separator.join([str(q) for q in quats])  # quats

        print(frame_string)

    def print_hierarchy(self, hierarchy: HierarchyDefinition, hierarchy_name: str):

        outstring = f"H {hierarchy_name} "
        outstring += self.separator.join(hierarchy.bone_names)
        print(outstring)

    def print_frame_debug(self, sample_idx, frame_idx, contacts, labels):

        debug_string = "A "  # Marker is always first element
        debug_string += f"Sample-index={sample_idx}/{self.sample_length-1}" + self.separator
        debug_string += f"Frame-index={frame_idx}/{self.total_samples-1}" + self.separator
        debug_string += f"Filename={self.data.get_filename_by_index(self.sample_idx)}" + self.separator
        debug_string += f"Contacts=[{','.join(['1' if x else '0' for x in contacts])}]" + self.separator
        debug_string += f"Labels={labels}"

        print(debug_string)


run()  # Encapsulate run behaviour to prevent globals

