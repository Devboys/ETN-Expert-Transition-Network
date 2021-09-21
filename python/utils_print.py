import numpy as np
import time
from etn_dataset import HierarchyDefinition

__all__ = ['print_hierarchy', 'print_sequences', 'print_single_sequence', 'print_frame']

def print_hierarchy(hierarchy: HierarchyDefinition, rig_name: str):
    """
    Prints formatted hierarchy info

    :param hierarchy: The hierarchy to print.
    :param rig_name: The chosen name of the hierarchy. Can be used to distinguish between multiple anim sequences.
    """

    joint_names = hierarchy.bone_names
    joint_offsets = hierarchy.bone_offsets
    parent_ids = hierarchy.parent_ids

    hierarchy_info = f"H {rig_name} {joint_names[0]}- "

    for j in range(1, len(joint_names)):  # we skip first parent id (root) because parent_ids[0] = -1
        parent_index = parent_ids[j]
        hierarchy_info += f"{joint_names[j]}-{joint_names[int(parent_index)]} "

    print(hierarchy_info)

    print(f"O {rig_name} {joint_names[0]} 0 0 0")
    for o in range(1, len(joint_offsets)):
        j_offset = joint_offsets[o]
        j_name = joint_names[o]
        offset_info = f"O {rig_name} {j_name} {j_offset[0]} {j_offset[1]} {j_offset[2]}"
        print(offset_info)


def print_sequences(
        sequences: np.ndarray,
        originals: np.ndarray,
        org_pos,
        pred_pos,
        joint_names,
        rig_name,
        frame_rate=15
):
    """

    :param sequences:
    :param originals:
    :param org_pos:
    :param pred_pos:
    :param joint_names:
    :param rig_name:
    :param frame_rate:
    :return:
    """
    s = 0
    while s < len(sequences):
        print_single_sequence(sequences[s], originals[s], org_pos[s], pred_pos[s], joint_names, frame_rate, rig_name)

        while True:
            message = input("Continue?\n")  # Newline makes sure unity reads the next message

            if message == "Next":
                s += 1
                break
            elif message == "Repeat":
                # Dont increment
                break
            elif message == "Prev":
                s -= 2
                break
            else:
                print("INCORRECT INPUT")
                continue
        # Bound s
        if s < 0:
            s = 0


def print_single_sequence(
        sequence: np.ndarray,
        original: np.ndarray,
        org_pos,
        pred_pos,
        joint_names,
        frame_rate: int,
        rig_name: str
):
    frame_time = (1 / frame_rate)

    print(f"B {len(sequence)}")

    for f in range(len(sequence)):
        start = time.perf_counter()

        # POSITIONS
        # print_frame(original[f], org_pos[f], joint_names, "original", positions=True)
        # print_frame(sequence[f], pred_pos[f], joint_names, rig_name, positions=True)

        # ROTATIONS
        print_frame(original[f], org_pos[f], joint_names, "original", positions=False)
        print_frame(sequence[f], pred_pos[f], joint_names, rig_name, positions=False)

        elapsed = time.perf_counter() - start
        time_left = frame_time - elapsed
        if time_left > 0:
            time.sleep(time_left)


def print_frame(frame, org_pos, joint_names, rig_name: str, positions=True):
    pos_l = 3 if positions else 0  # pos vector3
    rot_l = 4 if not positions else 0  # rot quaternion
    component_l = pos_l if positions else rot_l
    n_joint = (len(frame)) // component_l

    # root is a special case, always present
    r_q = frame[0:4]
    r_p = org_pos[0:3]

    print("G", rig_name, joint_names[0], r_p[0], r_p[1], r_p[2], r_q[1], r_q[2], r_q[3], r_q[0], flush=True)

    for j_id in range(1, n_joint):  # We already printed root joint, so start index at 1
        j_name = joint_names[j_id]
        idx = j_id * (pos_l + rot_l)
        q = frame[idx:idx + rot_l] if not positions else [1, 0, 0, 0]
        p = frame[idx + rot_l:idx + rot_l + pos_l] if positions else [0, 0, 0]
        print("G", rig_name, j_name, p[0], p[1], p[2], q[1], q[2], q[3], q[0])

    print(f"E {rig_name}")
