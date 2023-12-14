import numpy as np

HARTH_LABELS = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycking_sit",
    14: "cycling_stand",
    130: "cycling_sit_idle",
    140: "cycling_stand_idle"
}

harth_label_mapping = np.zeros(141)
harth_label_mapping[1:9] = np.arange(8)
harth_label_mapping[13] = 8
harth_label_mapping[14] = 9
harth_label_mapping[130] = 10
harth_label_mapping[140] = 11

UCI_HAR_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

ucihar_label_mapping = np.zeros(7)
ucihar_label_mapping[1:] = np.arange(6)