from torch.utils.data.dataset import IterableDataset
from copy import deepcopy
import numpy as np
import os
import glob
import tqdm
import random

from bvh_animation import BVHAnimation
from utils_fk import np_forward_kinematics_batch
from utils_norm import *


class ETNDataset(IterableDataset):
    def __init__(self,
                 data_dir: str,
                 train_data: 'ETNDataset' = None,
                 subsample_factor: int = 4,
                 joint_count=22
                 ):
        """
        An iterable, normalized dataset of animations of uniform length, read from BVH files. Data is formatted for
        compatibility with ETNGenerator. *MORE HERE*

         :param data_dir: Path to the directory containing the bvh files to read. Should NOT end with a '/'.
         :param train_data: *Optional* A dataset to extract normalization-parameters from. If None, norm-params will be
         calculated from this dataset. Use this when you want to synchronize normalization over multiple datasets.
         :param subsample_factor: The granularity factor, i.e. how much to subsample the dataset when generating
         samples.
         :param joint_count: The amount of joints in this dataset. TODO (2): check if this can be calculated from data
        """

        assert not data_dir.endswith("/"), "data_dir should not end with a '/'"

        self.joint_count = joint_count
        # Toebase and ankle joint indices for feet, should be adjusted to the used data set. For contact-calculations.
        self.leftFootJoint = [3, 4]
        self.rightFootJoint = [7, 8]
        # Velocity threshold value for contact-calculations
        self.velfactor = 0.02

        cache_path = data_dir + '/etn_cache.npz'
        if os.path.exists(cache_path):
            # Load data from cache
            with np.load(cache_path, allow_pickle=True) as data:
                self.joint_names = data["joint_names"]
                self.joint_offsets = data["joint_offsets"]
                self.animations = data["animations"]
        else:
            # Load data from files and cache
            bvh_paths = glob.glob(data_dir + "/*.bvh")
            assert len(bvh_paths) > 0, f"No .bvh files found in {data_dir}"
            # Get a bvh animation and extract hiearchy info from it. All files are assumed to use the same hierarchy.
            sample_anim = BVHAnimation(bvh_paths[0])
            self.joint_names = sample_anim.joints_names
            self.joint_offsets = sample_anim.joints_offsets
            # Load and format all bvh animations into input-vectors.
            self.animations = np.concatenate([
                self.to_etn_input(BVHAnimation(file), subsample_factor) for file in
                tqdm.tqdm(bvh_paths, desc=f"Loading bvh files from {data_dir}. This will only happen once.")
                ])
            np.savez_compressed(
                cache_path,
                animations=self.animations,
                joint_names=self.joint_names,
                joint_offsets=self.joint_offsets
            )

        # Resolve norm-params
        if train_data is None:
            roots = np.array([joint_r for joint_r in self.animations[:, 0]])
            self.root_norm_mean, self.root_norm_std = norm_params(roots)
        else:
            self.root_norm_mean = train_data.root_norm_std
            self.root_norm_std = train_data.root_norm_std

        # Normalize root velocities
        self.animations[:, 0] = normalize_vectors(self.animations[:, 0], self.root_norm_mean, self.root_norm_std)
        # Normalize root offsets
        self.animations[:, 2] = normalize_vectors(self.animations[:, 2], self.root_norm_mean, self.root_norm_std)

    def __iter__(self):
        while True:
            n_animations = len(self.animations)
            for idx in random.sample(range(n_animations), n_animations):
                root, quats, root_offsets, quat_offsets, target_frame, joint_offsets, parents, global_positions, \
                    contacts = self.animations[idx]

                ground_truth = np.concatenate([root[1:], quats[1:]], axis=1)
                yield root, quats, root_offsets, quat_offsets, target_frame, joint_offsets, parents.astype(int), \
                    ground_truth, global_positions, contacts

    def to_etn_input(self, animation: BVHAnimation, subsample_factor: int):
        """
        Process a BVHAnimation and format it into the following vectors (etn input format):
        :param animation: The BVH animation to process.
        :param subsample_factor: The granularity factor, i.e. how much to subsample the dataset when fetching animations
        :return: Returns a concatenated array of the processed vectors of length 10 + 32, overlapping by 15 frames.
        """

        # Define some basic vars
        past_length = 10        # Amount of frames in "past context".
        transition_length = 30  # Amount of frames in transition.
        target_length = 2       # Amount of frames in "target context".
        window_size = past_length + transition_length + target_length
        window_step = 15  # How many frames to step between each sample. Sample overlap if window step < size.

        processed_data = list()

        animation, joint_offsets, parents = animation.as_local_quaternions(subsample_factor)
        for window_index in range(1, len(animation), window_step):
            frames = animation[window_index: window_index + window_size]
            if len(frames) != window_size:
                continue  # Skip samples with too few frames (end samples)

            # Frame info vector(s)
            frames_copy = deepcopy(frames[1:])
            quats = frames_copy[:, 3:]
            root_vel = [frames[i][:3] - frames[i - 1][:3] for i in range(1, window_size)]

            # Offset vector(s)
            offsets = np.array([frames[i] - frames[-1] for i in range(0, window_size-1)])
            root_offsets = offsets[:, :3]
            quat_offsets = offsets[:, 3:]

            # Target vector(s)
            target_frame = frames[-1, 3:]  # Note: This is quats only.

            # Global join positions (through FK)
            global_positions = np_forward_kinematics_batch(
                np.repeat(joint_offsets.reshape([1, self.joint_count, 3]), window_size - 1, 0),
                np.concatenate([root_vel, quats], axis=1), parents, joint_count=self.joint_count)

            # Contacts
            pos = global_positions.reshape((41, 22, 3))  # TODO (2): base on global var instead of static "22"
            l_foot = (pos[1:, self.leftFootJoint, :] - pos[:-1, self.leftFootJoint, :]) ** 2
            r_foot = (pos[1:, self.rightFootJoint, :] - pos[:-1, self.rightFootJoint, :]) ** 2
            contacts_l = (np.sum(l_foot, axis=-1) < self.velfactor)
            contacts_r = (np.sum(r_foot, axis=-1) < self.velfactor)
            contacts = np.concatenate([contacts_l, contacts_r], axis=1)

            processed_data.append(np.array([
                root_vel,
                quats,
                root_offsets,
                quat_offsets,
                target_frame,
                joint_offsets,
                parents,
                global_positions[1:],
                contacts
            ], dtype=object))
        return np.array(processed_data)
