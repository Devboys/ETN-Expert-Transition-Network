from torch.utils.data.dataset import IterableDataset
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
                 past_length: int = 10,
                 transition_length: int = 30,
                 window_step: int = 15
                 ):
        """
        An iterable, normalized dataset of animations of uniform length, read from BVH files. Data is formatted for
        compatibility with ETNGenerator. *MORE HERE*

         :param data_dir: Path to the directory containing the bvh files to read. Should NOT end with a '/'.
         :param train_data: *Optional* A dataset to extract normalization-parameters from. If None, norm-params will be
         calculated from this dataset. Use this when you want to synchronize normalization over multiple datasets.
         samples.
        """

        assert not data_dir.endswith("/"), "data_dir should not end with a '/'"

        # Toebase and ankle joint indices for feet, should be adjusted to the used data set. For contact-calculations.
        self.lfoot_idx = [3, 4]
        self.rfoot_idx = [7, 8]
        # Velocity threshold value for contact-calculations
        self.velfactor = 1.7

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
                parsed_file = self.to_etn_input(bvh=BVHAnimation(file),
                                                past_length=past_length,
                                                transition_length=transition_length,
                                                window_step=window_step)
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

    def to_etn_input(self, bvh: BVHAnimation, past_length: int = 10, transition_length: int = 30, window_step: int = 15) -> np.ndarray:
        """
        Process a BVHAnimation and divide into samples of vectors of size (past_length + transition_length + 1).
            Each sample consists of a tuple of values for every frame:\n
            [0]=root joint velocity.\n
            [1]=pr-joint quaternion roations.\n
            [2]=root joint offsets from target frame.\n
            [3]=pr-joint quaternion offset from target frame.\n
            [4]=pr-joint quaternion rotations of target frame.\n
            [5]=global positions of every joint.\n
            [6]=contact tensors of feet joints.\n
            [7]=labels.

        :param bvh: The BVH animation to process.
        :param past_length: How many past-context frames to include in the sample.
        :param transition_length: How many ground-truth transition frames to include in the sample.
        :param window_step: Determines how many frames to step the sampling window between samples.
         A step less than (past_length + transition_length + 1) will result in overlapping samples.
        :return: An np.array of every sample extracted from the BVHAnimation file.
        """

        # Define some basic vars
        window_size = past_length + transition_length + 1  # TODO: make this one and remove -1 on all s_vars
        samples = list()
        subsample_factor = 4
        n_edge = 2  # num edge frames

        animation = bvh.as_local_quaternions(subsample_factor)

        root_vel = self.extract_root_velocities(animation)
        animation = animation[1:]  # skip first frame because root_vel cant be calculated

        quats = animation[:, 3:]

        global_positions = self.extract_glob_positions(animation)

        pos = global_positions.reshape((len(global_positions), self.hierarchy.bone_count(), 3))
        contacts = self.extract_feet_contacts(pos)

        labels = self.extract_labels(root_vel, pos, contacts, n_edge)

        for window_index in range(n_edge, len(animation)-n_edge, window_step):
            window_end_index = window_index + window_size
            frames = animation[window_index: window_end_index]
            if len(frames) != window_size:
                continue  # Skip samples with too few frames

            # Sample animation
            s_root_vel = root_vel[window_index: window_end_index]
            s_quats = quats[window_index: window_end_index]
            s_glob_pos = global_positions[window_index: window_end_index]
            s_contacts = contacts[window_index: window_end_index]
            s_labels = labels[window_index:window_end_index]

            # Sample offset vector(s)
            offsets = np.array([frames[i] - frames[-1] for i in range(0, window_size)])
            s_root_offsets = offsets[:, :3]
            s_quat_offsets = offsets[:, 3:]

            # Sample target vector(s)
            s_target_frame = frames[-1, 3:]  # Note: This is quats only.

            samples.append(np.array([
                s_root_vel,
                s_quats,
                s_root_offsets,
                s_quat_offsets,
                s_target_frame,
                s_glob_pos,
                s_contacts,
                s_labels
            ], dtype=object))
        return np.array(samples)

    def extract_root_velocities(self, frames) -> np.ndarray:
        root_vel = [frames[i][:3] - frames[i - 1][:3] for i in range(1, len(frames))]
        return np.asarray(root_vel)

    def extract_glob_positions(self, frames) -> np.ndarray:
        fk_offsets = np.repeat(self.hierarchy.bone_offsets.reshape([1, self.hierarchy.bone_count(), 3]),
                               len(frames), 0)
        fk_pose = frames  # concatenated (root_pos, joint_rots)
        global_positions = np_forward_kinematics_batch(
            offsets=fk_offsets,
            pose=fk_pose,
            parents=self.hierarchy.parent_ids,
            joint_count=self.hierarchy.bone_count()
        )

        return global_positions

    def extract_feet_contacts(self, pos) -> np.ndarray:
        """
        Extracts binary tensors of feet contacts

        :param pos: tensor of global joint positions of shape [n_frames, n_joints, 3]
        :return: binary tensors of (left foot contacts, right foot contacts) pr frame. Last frame is duplicated once.
        """

        vel_threshold = self.velfactor ** 2

        lfoot_xyz = (pos[1:, self.lfoot_idx, :] - pos[:-1, self.lfoot_idx, :]) ** 2
        contacts_l = (np.sum(lfoot_xyz, axis=-1) < vel_threshold)

        rfoot_xyz = (pos[1:, self.rfoot_idx, :] - pos[:-1, self.rfoot_idx, :]) ** 2
        contacts_r = (np.sum(rfoot_xyz, axis=-1) < vel_threshold)

        # Duplicate the last frame for shape consistency. Offset shouldn't be a problem because velocities are averages
        contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
        contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)
        contacts = np.concatenate((contacts_l, contacts_r), axis=1)

        return np.asarray(contacts)

    def extract_labels(self, root_vel, glob_pos, contacts, n_edge) -> np.ndarray:
        # ALL THRESHOLDS EXPECT SUBSAMPLE FACTOR OF 4. TODO: DO LABELING BEFORE SUBSAMPLE?

        # Extracts label tensors for every frame in the sample. Labeling is based on heuristics and is very unlikely to
        #  work for other datasets.

        n_labels = 4

        # when is hip considered moving
        max_idle_vel_standing = 4
        max_idle_vel_crouching = 2

        # locomotion thresholds
        max_vel_walk = 43
        # max_vel_run = 128

        crawl_height_thresh = 30

        # get feet velocity (sum of heel velocities)
        lheel_xyz = (glob_pos[1:, self.lfoot_idx[0], :] - glob_pos[:-1, self.lfoot_idx[0], :]) ** 2
        rheel_xyz = (glob_pos[1:, self.rfoot_idx[0], :] - glob_pos[:-1, self.rfoot_idx[0], :]) ** 2
        feet_vel = np.sqrt(np.sum(rheel_xyz, axis=1)) + np.sqrt(np.sum(lheel_xyz, axis=1))
        feet_vel = np.concatenate([feet_vel, feet_vel[-1:]], axis=0)  # duplicate last entry for shape consistency.

        # get root velocity
        root_xyz = root_vel ** 2
        ground_vel = np.sqrt(root_xyz[:, 0] + root_xyz[:, 2])

        labels = np.ndarray(shape=(len(root_vel), n_labels))
        for idx in range(n_edge, len(root_vel)-n_edge):
            # CRAWL LABEL
            neck_joint_idx = self.hierarchy.bone_names.index("Neck")
            dist_hip_neck = glob_pos[idx][neck_joint_idx] - glob_pos[idx][0]
            is_crawling = abs(dist_hip_neck[1]) < crawl_height_thresh  # y-component distance between hip->neck joints.

            max_mean_vel_idle = max_idle_vel_crouching if is_crawling else max_idle_vel_standing

            # IDLE LABEL
            # Is idle when average root velocity within n_edge of current frame is less than threshold
            mean_ground_vel = np.mean(ground_vel[idx - n_edge: idx + n_edge + 1])
            is_idle = mean_ground_vel < max_mean_vel_idle

            # LOCOMOTION LABELS
            # Is walking when no frames within n_edge of current frame has no contact with ground
            contacts_window = contacts[idx - n_edge: idx + n_edge + 1]
            contacts_ = [any(c for c in e) for e in contacts_window]
            is_walking = all(c == 1 for c in contacts_)
            is_walking = is_walking and not is_idle
            # Is running vs sprinting when not walking and any foot velocitys exceed given thresholds within n_edge
            #   of current frame.
            is_running = is_sprinting = False

            if not is_walking:
                feet_vel_window = feet_vel[idx - n_edge: idx + n_edge + 1]
                r = any(max_vel_walk < v for v in feet_vel_window)
                # s = any(max_vel_run < v for v in feet_vel_window)
                # if s:
                #     is_sprinting = True
                if r:
                    is_running = True

            if is_crawling:  # exclusive
                is_running = is_walking = False

            lab = [is_idle, is_walking, is_running, is_crawling]
            if all(e == 0 for e in lab):  # if no label, then idle default
                # TODO: consider an 'unlabeled' label and possibly an outlier-removal method.
                lab = [0]*n_labels
                lab[0] = True

            labels[idx] = lab

        return labels

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

    def vector_magn(self, vec) -> float:
        _sum = 0
        for e in vec:
            _sum += e ** 2

        return np.sqrt(_sum)
