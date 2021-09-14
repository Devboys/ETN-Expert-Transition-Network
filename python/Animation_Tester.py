import sys
import pathlib
from etn_dataset import ETNDataset
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
    data = ETNDataset(data_path, joint_count=22)
    player = SampleExplorer(sample_size=32, anim_dataset=data)
    player.play()


class PlaybackMode(Enum):
    SEQUENCE = 1
    FRAME = 2


class SampleExplorer:
    def __init__(self, sample_size, anim_dataset: ETNDataset):
        self.sample_size: int = sample_size
        self.data = anim_dataset
        self.mode = PlaybackMode.SEQUENCE
        self.running = True

    def global_to_actual(self, super_idx):
        sample_idx = int(super_idx / self.sample_size)
        frame_idx = super_idx % self.sample_size
        return [sample_idx, frame_idx]

    def actual_to_global(self, sample_idx, frame_idx):
        return sample_idx * self.sample_size + frame_idx

    def play(self):
        sample_idx = 0  # handles global sample index
        frame_idx = 0  # handles frame index in samples

        while self.running:
            message = input("Cont?\n")
            message = message.split("_")
            key = message[0]
            args = message[1:]

            if key == "NAV": # Navigate back & forward
                if args[0] == "NEXT":
                    sample_idx += 1
                elif args[0] == "PREV":
                    sample_idx -= 1

                sample = self.data.animations[sample_idx]
                self.print_sample(sample)

            elif key == "STOP": # Stop playback and close program
                self.running = False

            else:
                print("ERROR: could not parse message. Try again.\n")

    def print_sample(self, sample):
        for i in range(0, self.sample_size):

            self.print_frame(sample)

    def print_frame(self, frame):



run()  # Encapsulate run behaviour to prevent globals