import sys
import pathlib
from pathlib import Path
import uuid
from torch.utils.data import DataLoader
import numpy as np

import utils_print
from etn_model import ETNModel
from etn_generator import ETNGenerator
from etn_dataset import ETNDataset
from utils_print import *


def run(base_dir, is_param_optimizing: bool):
    # PATHS
    basedir = str(pathlib.Path(__file__).parent.parent.absolute())
    tensorboard_dir = f"{base_dir}/tensorboard/"
    model_dir = f"{base_dir}/models/"
    train_dir = f"{basedir}/data/lafan1_reduced/train"
    val_dir = f"{basedir}/data/lafan1_reduced/val"

    model_id = str(uuid.uuid4())[:8] if is_param_optimizing else "NaN"

    # HYPERPARAMS
    minibatch_size = 32
    n_epochs = 11  # i.e. training length
    learning_rate = 0.0005

    if is_param_optimizing:
        # for hyperparam optimization, learning rate is a random float between 0.0001 and 0.1
        power = -3 * np.random.rand()
        learning_rate = 10**(-1 + power)

    model_name = f"etn_{model_id}_bs{str(minibatch_size)}_ne{str(n_epochs)}_lr{str(learning_rate)}.pt"
    model_path = model_dir + model_name

    train_data = ETNDataset(train_dir)
    train_loader = DataLoader(train_data, batch_size=minibatch_size)
    val_data = ETNDataset(val_dir, train_data=train_data)
    val_loader = DataLoader(val_data, batch_size=minibatch_size)

    model = ETNModel(hierarchy=val_data.hierarchy, learning_rate=learning_rate, name=model_name)

    if Path(model_path).exists():
        # Load model
        model.load(model_path)
        print(f"Loaded model: {model_path}")

        # Get next batch and extract
        batch = next(iter(val_loader))
        batch = [b.float().to(model.device) for b in batch]  # Convert to float values for concat
        root = batch[0]
        quats = batch[1]
        root_offsets = batch[2]
        quat_offsets = batch[3]
        target_quats = batch[4]
        ground_truth = batch[5]
        global_positions = batch[6]
        contacts = batch[7]
        labels = batch[8]

        # Predict transition
        pred_poses, pred_contacts = model.forward(
            past_root=root[:, :10],
            past_quats=quats[:, :10],
            past_root_offset=root_offsets[:, :10],
            past_quat_offset=quat_offsets[:, :10],
            past_contacts=contacts[:, :10],
            target_quats=target_quats,
            target_root_pos=-root_offsets[:, 0]
        )
        pred_pos = model.fk(pred_poses)

        # Parse 'original' data aka batch. Used for comparison
        root, quats, root_offsets, quat_offsets, target_quats, ground_truth, global_positions, \
            contacts, labels = [b.float().to(model.device) for b in batch]

        org_quats, pred_quats, org_pos, pred_pos, org_contacts, pred_contacts = utils_print.process_sample_pair(quats,
                                                                                                    pred_poses,
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
        model.do_train(train_loader, val_loader, n_epochs, 10, tensorboard_dir)
        model.save(model_path)


run(sys.path[0], False)  # Encapsulate run behaviour to prevent globals

