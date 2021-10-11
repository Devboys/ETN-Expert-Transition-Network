from bvh import Bvh
import numpy as np
import transformations as transform

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

