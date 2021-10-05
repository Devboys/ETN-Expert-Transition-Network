from bvh import Bvh
import numpy as np
import transformations as transform
import math


class BVHAnimation:
    ## missing support to channel order, so that it can automatically convert any euler angle order
    def __init__(self, bvh_path, scale_factor: float = 1.0):
        with open(bvh_path) as f:
            mocap = Bvh(f.read())

        self.joints_names = mocap.get_joints_names()
        self.n_joints = len(self.joints_names)
        self.joints_parent_ids = np.array([mocap.joint_parent_index(j_name) for j_name in self.joints_names])
        self.joints_channel_ids = np.array([mocap.get_joint_channels_index(j_name) for j_name in self.joints_names])
        self.joints_offsets = np.array([mocap.joint_offset(j_name) for j_name in self.joints_names])
        self.frames = np.array([np.array(xi, float) for xi in mocap.frames])
        self.frames[:, 3:] *= np.pi / 180  # rotation angles from degrees to radians
        self.frame_time = mocap.frame_time
        self.n_frames = mocap.nframes
        self.scale_factor = scale_factor

    def get_pose(self, frame, local_space=False):
        pose = [None] * len(self.joints_names)

        for id in range(self.n_joints):
            channel = self.joints_channel_ids[id]
            parent_id = self.joints_parent_ids[id]

            if parent_id < 0:  # is root
                mat = transform.euler_matrix(self.frames[frame, channel + 5],
                                             self.frames[frame, channel + 4],
                                             self.frames[frame, channel + 3], 'sxyz')  # rotation
                # mat[0:3, 3] = self.frames[frame, channel:channel + 3]  # translation
                mat[0, 3] = self.frames[frame, channel + 0] * self.scale_factor
                mat[1, 3] = self.frames[frame, channel + 1] * self.scale_factor
                mat[2, 3] = self.frames[frame, channel + 2] * self.scale_factor
                pose[id] = mat

            else:
                mat = transform.euler_matrix(self.frames[frame, channel + 2],
                                             self.frames[frame, channel + 1],
                                             self.frames[frame, channel], 'sxyz') # Rotation
                # mat[0:3, 3] = self.joints_offsets[id]   # translation
                mat[0, 3] = self.joints_offsets[id, 0] * self.scale_factor
                mat[1, 3] = self.joints_offsets[id, 1] * self.scale_factor
                mat[2, 3] = self.joints_offsets[id, 2] * self.scale_factor

                pose[id] = pose[id] = mat if local_space else np.dot(pose[parent_id], mat)  # if global space, resolve parent-child relative transform

        return pose

    def get_animation_global(self, subsamplestep=1):
        nframes = math.floor((self.n_frames + subsamplestep - 1) / subsamplestep)
        poses = [None] * nframes
        current = 0
        for frame in range(0, self.n_frames, subsamplestep):
            poses[current] = self.get_pose(frame)
            current += 1

        return poses

    def get_animation_local(self, subsamplestep=1):
        nframes = math.floor((self.n_frames + subsamplestep - 1) / subsamplestep)
        poses = [None] * nframes
        current = 0
        for frame in range(0, self.n_frames, subsamplestep):
            poses[current] = self.get_pose(frame, True)
            current += 1

        return poses

    def as_joint_position(self, subsamplestep=1):
        # poses is a list (frames) of lists (joints), with a 4x4 transformation matrix in each entry
        # output is a pose per row:
        # root quaternion - root positions - joint positions ...

        poses = self.get_animation_root_space(subsamplestep)
        anim_vec = np.zeros((len(poses), 4 + len(poses[0]) * 3))
        for p in range(len(poses)):
            # we assume that the first entry is the root
            mat = poses[p][0]
            anim_vec[p, 0:4] = transform.quaternion_from_matrix(mat, isprecise=True)
            anim_vec[p, 4:7] = transform.translation_from_matrix(mat)
            col = 7
            for j in range(1, len(poses[p])):
                j_mat = poses[p][j]
                anim_vec[p, col:col + 3] = transform.translation_from_matrix(j_mat)
                col += 3

        return anim_vec

    def get_animation_root_space(self, subsamplestep=1):
        nframes = math.floor((self.n_frames + subsamplestep - 1) / subsamplestep)
        poses = [None] * nframes
        current = 0
        for frame in range(0, self.n_frames, subsamplestep):
            poses[current] = self.get_pose(frame)
            root_inv = transform.inverse_matrix(poses[current][0])
            for j_id in range(1, self.n_joints):
                poses[current][j_id] = np.dot(root_inv, poses[current][j_id])
            current += 1

        return poses

    def get_animation_minus_root_pos(self, subsamplestep=1):
        nframes = math.floor((self.n_frames + subsamplestep - 1) / subsamplestep)
        poses = [None] * nframes
        current = 0
        for frame in range(0, self.n_frames, subsamplestep):
            poses[current] = self.get_pose(frame)
            root_pos = poses[current][0][0:3, 3]
            for j_id in range(1, self.n_joints):
                poses[current][j_id][0:3, 3] -= root_pos
            current += 1

        return poses

    def as_root_relative_pos(self, subsamplestep=1, joint_subset = None):
        # poses is a list (frames) of lists (joints), with a 4x4 transformation matrix in each entry
        # output is a pose per row:
        # root position - joint positions ...

        if joint_subset is None:
            joint_subset = range(self.n_joints)
        poses = self.get_animation_global(subsamplestep)
        anim_vec = np.zeros((len(poses), len(joint_subset) * 3))

        for p in range(len(poses)):
            # we assume that the first entry is the root
            root_mat = poses[p][0]
            anim_vec[p, 0:3] = transform.translation_from_matrix(root_mat)
            col = 3
            for j in range(1, len(joint_subset)):
                j = joint_subset[j]
                j_mat = poses[p][j]
                anim_vec[p, col:col + 3] = transform.translation_from_matrix(j_mat) - anim_vec[p, 0:3]  # root space = subtract root pos AKA root as origin
                col += 3
        return anim_vec

    def as_local_quaternions(self, subsamplestep=1):
        new_frame_count = self.n_frames//subsamplestep
        anim_vec = np.zeros((new_frame_count, self.n_joints*4+3))

        for frame_index in range(0, new_frame_count):
            frame = self.frames[frame_index*subsamplestep]
            anim_vec[frame_index, :3] = frame[:3]  # First three elements in pose is root joint position.
            for joint_index in range(self.n_joints):
                e_index = joint_index * 3 + 3  # BVH stores rotations as euler angles
                q_index = joint_index * 4 + 3  # We want to store as quaternions
                euler = frame[e_index:e_index+3]  # in rads
                quat = transform.quaternion_from_euler(euler[0], euler[1], euler[2], "rzyx")  # output order is [WXYZ]
                anim_vec[frame_index, q_index:q_index+4] = quat
        return anim_vec

    def as_local_euler(self, subsamplestep=1):
        new_frame_count = self.n_frames // subsamplestep
        anim_vec = np.zeros((new_frame_count, self.n_joints * 3 + 3))

        for frame_index in range(0, new_frame_count):
            frame = self.frames[frame_index * subsamplestep]
            anim_vec[frame_index, :3] = frame[:3]  # First three elements in pose is root joint position.
            for joint_index in range(self.n_joints):
                e_index = joint_index * 3 + 3  # BVH stores rotations as euler angles
                euler = frame[e_index:e_index + 3] * (180/np.pi)  # rots stored as rads. Convert to degrees
                anim_vec[frame_index, e_index:e_index + 3] = euler
        return anim_vec

    def joint_name_subset(self, subset_indices):
        joint_list = np.empty(len(subset_indices), dtype='U20')  # U20 -> unicode str of 20 characters
        for j in range(0, len(subset_indices)):
            joint_list[j] = str(self.joints_names[subset_indices[j]])

        return joint_list

    def parent_ids_subset(self, subset_indices):
        parent_ids = np.empty(len(subset_indices), dtype=int)
        for j in range(0, len(subset_indices)):
            parent_ids[j] = str(self.joints_parent_ids[subset_indices[j]])

        return parent_ids

    def convert_parent_ids(self, parent_indices, subset_indices: list):
        converted_ids = np.empty(len(parent_indices), dtype=int)
        for p in range(0, len(converted_ids)):
            if parent_indices[p] == -1:
                continue
            if parent_indices[p] in subset_indices:
                converted_ids[p] = subset_indices.index(parent_indices[p])
            else:
                valid_parent_id = self.get_valid_parent_id(parent_indices[p], subset_indices)
                converted_ids[p] = subset_indices.index(valid_parent_id)

        return converted_ids

    def get_valid_parent_id(self, current_index, subset_indices):
        if current_index == -1:
            return current_index
        elif current_index in subset_indices:
            return current_index
        else:
            return self.get_valid_parent_id(self.joints_parent_ids[current_index], subset_indices)

