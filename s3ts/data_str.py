from dataclasses import dataclass

@dataclass
class TaskParameters:

    """ Settings for the auxiliary tasks. """

    main_weight: int = 3
    
    disc: bool = True               # Discretized clasification
    discrete_intervals: int = 5     # Discretized intervals
    disc_weight: int = 1

    pred: bool = True               # Prediction
    pred_time: int = None           # Prediction time (if None, then window_size)
    pred_weight: int = 1

    auto: bool = True               # Auto prediction

    areg_ts: bool = True            # Time-Series Autoregression
    areg_ts_weight: int = 1

    areg_img: bool = True           # Similarity Frame Autoregression
    areg_img_weight: int = 1

@dataclass
class AugProbabilities:
    """ Data augmentation probabilites. """
    jitter:      float = 0
    scaling:     float = 0
    time_warp:   float = 0
    window_warp: float = 0
