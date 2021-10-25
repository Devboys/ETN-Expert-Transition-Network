from collections import deque
import torch as t
import numpy as np

__all__ = ['np_forward_kinematics_batch', 'torch_forward_kinematics_batch']


def np_forward_kinematics_batch(offsets, pose, parents, joint_count=22):
    """
    Computes the final positions and rotations in global space of joints in a skeleton hierarchy.

    :param offsets: Batch of shape (batch_size, num_joints*3) of local offsets of each joint o
    :param pose: Batch of shape (batch_size, num_joints*4+3) of Vectors containing the root joint position and a joints' local quaternions
    :param parents: One-dimensional vector representing the mapping from a joint to its parent
    :param joint_count: The amount of joints in the hierarchy
    :return: Returns a vector containing the final joint positions and quaternions.
    """

    def get_children(joint_index):
        return [c for c in range(len(parents)) if c != joint_index and parents[c] == joint_index]

    def get_conjugate(q):
        return np.concatenate([q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]], axis=1)

    def mul(q, p):
        qv = q[:, 1:]  # (qx, qy, qz)
        pv = p[:, 1:]  # (px, py, pz)
        qw = q[:, :1]
        pw = p[:, :1]
        rv = pw * qv + qw * pv + np.cross(qv, pv)
        rs = pw * qw - np.matmul(pv, np.transpose(qv)).diagonal().reshape((q.shape[0], 1))

        return np.concatenate([rs, rv], axis=1)

    def mulv(q, v):
        zero = np.zeros([v.shape[0], 1], dtype=np.float32)
        p = np.concatenate([zero, v[:, :1], v[:, 1:2], v[:, 2:3]], axis=1)
        return mul(mul(q, p), get_conjugate(q))[:, 1:]

    joints = deque()
    joints.append(0)
    global_positions = [None] * joint_count
    global_rotations = [None] * joint_count
    global_positions[0] = pose[:, :3]
    global_rotations[0] = pose[:, 3:7]
    while len(joints) > 0:
        joint = joints.pop()
        for child in get_children(joint):
            joints.append(child)
            q_index = joint * 4 + 3
            cq_index = child * 4 + 3
            j_pos = global_positions[joint]
            j_rot = global_rotations[joint]
            c_pos = offsets[:, child]
            c_rot = pose[:, cq_index:cq_index + 4]

            c_pos = mulv(j_rot, c_pos) + j_pos
            c_rot = mul(j_rot, c_rot)

            global_positions[child] = c_pos
            global_rotations[child] = c_rot
    return np.concatenate(global_positions, axis=1)


def torch_forward_kinematics_batch(offsets, pose, parents, joint_count):
    """
    Computes the global positions and rotations in global space of joints in a hierarchy. \n
    :param offsets: Batch of shape (batch_size, num_joints*3) of local offsets of each joint o
    :param pose: Batch of shape (batch_size, num_joints*4+3) of Vectors containing the root joint position and a joints' local quaternions
    :param parents: One-dimensional vector representing the mapping from a joint to its parent
    :param joint_count: The amount of joints in the hierarchy.
    :return: Returns a vector containing the final joint positions and quaternions.
    """
    def get_children(joint_index) -> list:
        """
        Rerturns indices of all child-joints of the joint at given index.
        :return: List of child indices.
        """
        return [c for c in range(len(parents)) if c != joint_index and parents[c] == joint_index]

    # Get conjugate of a quaternion (batch)
    def get_conjugate(q): return t.cat([q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]], dim=1)

    def mul(q, p):
        """
        Multiplies two quaternions \n
        :param q: First quaternion
        :param p: Second quaternion
        """
        qv = q[:, 1:]  # (qx, qy, qz)
        pv = p[:, 1:]  # (px, py, pz)
        qw = q[:, :1]
        pw = p[:, :1]
        rv = pw * qv + qw * pv + t.cross(qv, pv)
        rs = pw * qw - t.matmul(pv, t.transpose(qv, 0, 1)).diagonal().reshape((q.shape[0], 1))
        return t.cat([rs, rv], dim=1)

    # Quaternion-vector multiplication From lafan1 repo
    def mulv(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        q = q.view(-1, 4)
        v = v.view(-1, 3)

        qvec = q[:, 1:]
        uv = t.cross(qvec, v, dim=1)
        uuv = t.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    joints = deque()
    joints.append(0)
    global_positions = [None] * joint_count
    global_rotations = [None] * joint_count
    global_positions[0] = pose[:, :3]
    global_rotations[0] = pose[:, 3:7]

    # Starting from root, visit each joint and apply the parents transformation to the child
    # Multiplies the rotation quaternions and rotates the joint offset vector (offset from the parent), then adds the
    # parent position to get the global position
    while len(joints) > 0:
        joint = joints.pop()
        for child in get_children(joint):
            joints.append(child)
            q_index = joint * 4 + 3
            cq_index = child * 4 + 3
            j_pos = global_positions[joint]
            j_rot = global_rotations[joint]
            c_pos = offsets[:, child]
            c_rot = pose[:, cq_index:cq_index+4]

            c_pos = mulv(j_rot, c_pos) + j_pos
            c_rot = mul(j_rot, c_rot)

            global_positions[child] = c_pos
            global_rotations[child] = c_rot
    return t.cat(global_positions, dim=1)