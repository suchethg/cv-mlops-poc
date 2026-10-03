"""CholecSeg8k label definitions.

Watershed masks store each pixel's class as a gray value. These are the 13
classes from the CholecSeg8k paper, mapped to training IDs 0-12.
"""
GRAY_TO_CLASS = {
    50: (0, "background"),
    11: (1, "abdominal_wall"),
    21: (2, "liver"),
    13: (3, "gastrointestinal_tract"),
    12: (4, "fat"),
    31: (5, "grasper"),
    23: (6, "connective_tissue"),
    24: (7, "blood"),
    25: (8, "cystic_duct"),
    32: (9, "l_hook_electrocautery"),
    22: (10, "gallbladder"),
    33: (11, "hepatic_vein"),
    5: (12, "liver_ligament"),
}
CLASS_NAMES = [name for _, (_, name) in sorted(GRAY_TO_CLASS.items(), key=lambda kv: kv[1][0])]