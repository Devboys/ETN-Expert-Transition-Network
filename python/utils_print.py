from etn_dataset import HierarchyDefinition
from enum import Enum

__all__ = ['PlaybackMode','print_hierarchy', 'print_frame']


class PlaybackMode(Enum):
    SEQUENCE = 1
    FRAME = 2


def print_hierarchy(hierarchy: HierarchyDefinition, hierarchy_name: str, separator: str):
    out_string = f"H {hierarchy_name} "
    out_string += separator.join(hierarchy.bone_names)
    print(out_string)


def print_frame(root_pos, quats, rig_name: str, separator: str):
    """
    Prints a single frame of animation in the format "P <rig_name> <root_pos><separator><quats>".
    Note that message marker and rig name are separated with a whitespace, while other elements use the provided separator.

    :param root_pos: The global root position of the pose
    :param quats: The quaternion rotations of each joint in the pose
    :param rig_name: The name of the rig. (Can be used to run multiple animations at the same time)
    :param separator: The separator to use between message elements
    """

    frame_string = f"P {rig_name} "  # frame marker + rig identifier
    # frame_string = frame_string + separator.join([str(p) for p in root_pos])  # root position
    # frame_string += separator  # separator
    frame_string += separator.join([str(q) for q in quats])  # quats

    print(frame_string)
