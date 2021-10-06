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


class HierarchyDefinition:
    def __init__(self, bone_names, bone_offsets, parent_ids):
        """
        A simple data object to contain hierarchy-level information with some simple helper classes.

        :param bone_names: tuple of bone names.
        :param bone_offsets: tuple of bone offsets.
        :param parent_ids: tuple of parent indices for bones.
        """

        self.bone_names = bone_names
        self.bone_offsets = bone_offsets
        self.parent_ids = parent_ids

    def bone_count(self):
        return len(self.bone_names)


class ETNDataset(IterableDataset):
    def __init__(self,
                 data_dir: str,
                 train_data: 'ETNDataset' = None,
                 subsample_factor: int = 4
                 ):
        """
        An iterable, normalized dataset of animations of uniform length, read from BVH files. Data is formatted for
        compatibility with ETNGenerator. *MORE HERE*

         :param data_dir: Path to the directory containing the bvh files to read. Should NOT end with a '/'.
         :param train_data: *Optional* A dataset to extract normalization-parameters from. If None, norm-params will be
         calculated from this dataset. Use this when you want to synchronize normalization over multiple datasets.
         :param subsample_factor: The granularity factor, i.e. how much to subsample the dataset when generating
         samples.
        """

        assert not data_dir.endswith("/"), "data_dir should not end with a '/'"

        # Toebase and ankle joint indices for feet, should be adjusted to the used data set. For contact-calculations.
        self.lfoot_idx = [3, 4]
        self.rfoot_idx = [7, 8]
        # Velocity threshold value for contact-calculations
        self.velfactor = 0.02

        cache_path = data_dir + '/etn_cache.npz'
        if os.path.exists(cache_path):
            # Load data from cache
            with np.load(cache_path, allow_pickle=True) as data:
                self.animations = data["animations"]
                self.hierarchy = HierarchyDefinition(data["bone_names"], data["bone_offsets"], data["parent_ids"])
                self.file_indices = data["file_indices"]
        else:
            # Load data from files and cache
            bvh_paths = glob.glob(data_dir + "/*.bvh")
            assert len(bvh_paths) > 0, f"No .bvh files found in {data_dir}"
            # Get a bvh animation and extract hierarchy info from it. All files are assumed to use the same hierarchy.
            sample_anim = BVHAnimation(bvh_paths[0])
            self.hierarchy = HierarchyDefinition(bone_names=sample_anim.joints_names,
                                                 bone_offsets=sample_anim.joints_offsets,
                                                 parent_ids=sample_anim.joints_parent_ids
                                                 )
            anims = list()
            file_start_idx = 0
            file_end_idx = -1
            self.file_indices = list()
            # Load all bvh files and format animations into expected input format.
            for file in tqdm.tqdm(bvh_paths, desc=f"Loading bvh files from {data_dir}. This will only happen once."):
                parsed_file = self.to_etn_input(BVHAnimation(file), subsample_factor, 42)
                anims.append(parsed_file)

                file_end_idx += len(parsed_file)
                file_name = os.path.basename(file)
                self.file_indices.append((file_name, file_start_idx, file_end_idx))

                file_start_idx = file_end_idx + 1

            self.animations = np.concatenate(anims)

            # np.savez_compressed( # TODO: REVERT
            #     cache_path,
            #     animations=self.animations,
            #     bone_offsets=self.hierarchy.bone_offsets,
            #     bone_names=self.hierarchy.bone_names,
            #     parent_ids=self.hierarchy.parent_ids,
            #     file_indices=self.file_indices
            # )

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
                root, quats, root_offsets, quat_offsets, target_frame, global_positions, contacts, \
                    labels = self.animations[idx]

                ground_truth = np.concatenate([root, quats], axis=1)

                # TODO: Add labels as output of iterator and include in generator code
                yield root, quats, root_offsets, quat_offsets, target_frame, ground_truth, global_positions, contacts

    def to_etn_input(self, animation: BVHAnimation, subsample_factor: int, window_step: int = 15):
        """
        Process a BVHAnimation and format it into the following vectors (etn input format):
        :param animation: The BVH animation to process.
        :param subsample_factor: The granularity factor, i.e. how much to subsample the dataset when fetching animations
        :param window_step: Determines how many frames to step the sampling window between samples.
        A step less than 41 will result in overlapping samples.
        :return: Returns a concatenated array of the processed vectors of length 10 + 32.
        """

        # Define some basic vars
        past_length = 10        # Amount of frames in "past context".
        transition_length = 30  # Amount of frames in transition.
        target_length = 2       # Amount of frames in "target context".
        window_size = past_length + transition_length + target_length

        processed_data = list()

        animation = animation.as_local_quaternions(10)
        for window_index in range(0, len(animation), window_step):
            frames = animation[window_index: window_index + window_size]
            if len(frames) != window_size:
                continue  # Skip samples with too few frames

            # Frame info vector(s)
            frames_copy = deepcopy(frames[1:])
            quats = frames_copy
            root_vel = [frames[i][:3] - frames[i - 1][:3] for i in range(1, window_size)]

            # Offset vector(s)
            offsets = np.array([frames_copy[i] - frames_copy[-1] for i in range(0, window_size-1)])
            root_offsets = offsets[:, :3]
            rot_offsets = offsets[:, 3:]

            # Target vector(s)
            target_frame = frames_copy[-1, 3:]  # Note: This is rotation only.

            fk_offsets = np.repeat(self.hierarchy.bone_offsets.reshape([1, self.hierarchy.bone_count(), 3]), window_size-1, 0)
            fk_pose = np.concatenate([root_vel, quats], axis=1)
            # Global joint positions (through FK)
            global_positions = np_forward_kinematics_batch(
                offsets=fk_offsets,
                pose=fk_pose,
                parents=self.hierarchy.parent_ids,
                joint_count=self.hierarchy.bone_count())

            # Contacts
            pos = global_positions.reshape((window_size-1, self.hierarchy.bone_count(), 3))
            contacts = self.extract_feet_contacts(pos, self.lfoot_idx, self.rfoot_idx, self.velfactor)
            contacts = np.concatenate(contacts, axis=1)

            # Autolabel frames.
            labels = self.extract_labels(root_vel, global_positions[:, :3])

            processed_data.append(np.array([
                root_vel,
                quats,
                root_offsets,
                rot_offsets,
                target_frame,
                global_positions,
                contacts,
                labels
            ], dtype=object))
        return np.array(processed_data)

    def extract_feet_contacts(self, pos, lfoot_idx, rfoot_idx, vel_threshold=0.02):
        """
        Extracts binary tensors of feet contacts

        :param pos: tensor of global joint positions of shape [n_frames, n_joints, 3]
        :param lfoot_idx: hierarchy indices of left foot joints (heel_idx, toe_idx)
        :param rfoot_idx: hierarchy indices of right foot joints (heel_idx, toe_idx)
        :param vel_threshold: velocity threshold to consider a joint moving or not.
        :return: binary tensors of (left foot contacts, right foot contacts) pr frame. Last frame is duplicated once.
        """

        lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
        contacts_l = (np.sum(lfoot_xyz, axis=-1) < vel_threshold)

        rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
        contacts_r = (np.sum(rfoot_xyz, axis=-1) < vel_threshold)

        # Duplicate the last frame for shape consistency. Offset shouldn't be a problem because velocities are averages
        contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
        contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

        return np.asarray(contacts_l), np.asarray(contacts_r)

    def extract_labels(self, root_vel, root_pos):
        # TODO: actual labeling.
        # If root joint velocity is less than threshold, then label = idle, else label = moving
        # vel_thresh = 0.02  # EXAMPLE
        # height_thresh = 2  # EXAMPLE
        # moving_labels = [1 if vel > vel_thresh else 0 for vel in root_vel]  # EXAMPLE
        #
        # # If root joint position (height) lower than threshold, then label = crouching
        # standing_labels = [1 if pos > height_thresh else 0 for pos in root_pos]  # EXAMPLE
        # labels = np.concatenate(moving_labels, standing_labels, axis=1)

        return np.tile([0, 1, 0, 1], (len(root_vel), 1))  # placeholder

    def get_filename_by_index(self, idx) -> str:
        """
        Returns the filename of the sample at the given index. Returns NONE if no such index.
        """

        filename = "NONE"
        for file in self.file_indices:
            if int(file[1]) <= idx <= int(file[2]):
                filename = file[0]

        return filename

    def get_index_of_filename(self, filename: str):
        """
        Returns the start and end sample indices of the given file

        :return: A tuple of (start_index, end_index). If filename is not found, will return (-1, -1)
        """
        start_idx = -1
        end_idx = -1

        for file in self.file_indices:
            if filename == file[0]:
                start_idx = int(file[1])
                end_idx = int(file[2])

        return start_idx, end_idx
