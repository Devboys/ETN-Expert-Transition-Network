import datetime
import os
from torch.utils.tensorboard import SummaryWriter

__all__ = ['get_writers']


def get_writers(base_path, subfolder_name):
    """
    Creates two tensorboard Summary Writers for training and validation loss, respectively.
    Logs can be found in the tensorboard folder and are separated into folder names given by subfolder_name.

    :param base_path: The base folder to save the tensorboard log folders to
    :param subfolder_name: The sub-folder to save logs to. Will be appended with datetime & error message
    :return: The created training and validation writers
    """
    tensorboard_dir = base_path
    revision = os.environ.get("REVISION") or datetime.datetime.now().strftime("%m%d-%H%M")
    message = os.environ.get('MESSAGE')

    train_writer = SummaryWriter(tensorboard_dir + f'{revision}-{subfolder_name}/train/{message}')
    val_writer = SummaryWriter(tensorboard_dir + f'{revision}-{subfolder_name}/val/{message}')
    return train_writer, val_writer
