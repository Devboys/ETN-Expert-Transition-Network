import sys
import pathlib
from pathlib import Path
import uuid
from torch.utils.data import DataLoader
import numpy as np

from etn_generator import ETNGenerator
from etn_dataset import ETNDataset
from utils_print import print_hierarchy, print_sequences


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

    train_data = ETNDataset(f"{basedir}/data/lafan1_reduced/train", joint_count=num_joints)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = ETNDataset(f"{basedir}/data/lafan1_reduced/val", joint_count=num_joints, train_data=train_data)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    generator = ETNGenerator(
        learning_rate=learning_rate,
        num_joints=num_joints,
        use_gan=False
    )

    if Path(model_path).exists():
        # Load model
        generator.load(model_path)
        print(f"Loaded model: {model_path}")

        # Get generated transition
        batch = next(iter(val_loader))
        pred_positions, _, pred_quats = generator.eval_batch(batch)

        # Parse 'original' data aka batch. Used for comparison
        root, quats, root_offsets, quat_offsets, target_quats, joint_offsets, parents, ground_truth, global_positions, \
            contacts = [b.float().to(generator.device) for b in batch]
        org_quats = quats.detach().cpu().numpy()
        org_quats = org_quats[:, 10:]  # Trim past-context frames
        org_root_pos = global_positions.detach().cpu().numpy()
        org_root_pos = org_root_pos[:, 10:, :3]  # Trim past-context & quats

        # Parse 'predicted' data aka generated transition.
        pred_quats = pred_quats.detach().cpu().numpy()
        pred_quats = pred_quats[:, :, 3:]  # TODO: ensure this is correct. Old note: cut out root pos
        pred_root_pos = pred_positions.detach().cpu().numpy()
        pred_root_pos = pred_root_pos[:, :, :3]

        # Print hierarchy info
        print_hierarchy(val_data.joint_names, parents, val_data.joint_offsets, "original")
        print_hierarchy(val_data.joint_names, parents, val_data.joint_offsets, "prediction")

        print_sequences(pred_quats, org_quats, org_root_pos, pred_root_pos, val_data.joint_names, "prediction", 30)
    else:
        generator.do_train(train_loader, model_id, tensorboard_dir, n_batches, val_loader)
        generator.save(model_path)


run(sys.path[0], True)  # Encapsulate run behaviour to prevent globals
