import datetime
import os
from torch.utils.tensorboard import SummaryWriter

__all__ = ['get_writers']


def get_writers(subfolder_name):
    """
    Creates two tensorboard Summary Writers for training and validation loss, respectively.
    Logs can be found in the tensorboard folder and are separated into folder names given by subfolder_name.

    :param subfolder_name: The sub-folder to save logs to. Will be appended with datetime & error message
    :return: The created training and validation writers
    """
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or 'tensorboard'
    revision = os.environ.get("REVISION") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    message = os.environ.get('MESSAGE')

    train_writer = SummaryWriter(tensorboard_dir + f'/{subfolder_name}-{revision}/train/{message}')
    val_writer = SummaryWriter(tensorboard_dir + f'/{subfolder_name}-{revision}/val/{message}')
    return train_writer, val_writer
