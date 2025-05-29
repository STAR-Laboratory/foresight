import torch
from videosys.utils.logging import logger

FORESIGHT_MANAGER = None


class FORESIGHTConfig:
    def __init__(
        self,
        warmup: int = None,
        recalculate: int = None,
        threshold: float = None,
    ):
        self.steps = None

        self.warmup = warmup
        self.recalculate = recalculate
        self.threshold = threshold

class FORESIGHTManager:
    def __init__(self, config: FORESIGHTConfig):
        self.config: FORESIGHTConfig = config

        init_prompt = f"Init Foresight."
        init_prompt += f" Warmup Steps: {config.warmup}."
        init_prompt += f" Recalculate: {config.recalculate}."
        init_prompt += f" Threshold: {config.threshold}."
        logger.info(init_prompt)


def set_foresight_manager(config: FORESIGHTConfig):
    global FORESIGHT_MANAGER
    FORESIGHT_MANAGER = FORESIGHTManager(config)


def enable_foresight():
    if FORESIGHT_MANAGER is None:
        return False
    return True

def foresight_layer_skip(spatial_mse, spatial_threshold, temporal_mse, temporal_threshold):
    spatial_binary = (spatial_mse > spatial_threshold).int()
    spatial_zero_idx = torch.where(spatial_binary == 0)[0]
    spatial_last_zer0_idx = spatial_zero_idx[-1].item() if len(spatial_zero_idx) > 0 else -1

    temporal_binary = (temporal_mse > temporal_threshold).int()
    temporal_zero_idx = torch.where(temporal_binary == 0)[0]
    temporal_last_zer0_idx = temporal_zero_idx[-1].item() if len(temporal_zero_idx) > 0 else -1

    layer_idx = max(spatial_last_zer0_idx, temporal_last_zer0_idx)
    return layer_idx

def foresight_layer_skip_spatial(spatial_mse, spatial_threshold):
    spatial_binary = torch.tensor(spatial_mse) > torch.tensor(spatial_threshold)
    spatial_zero_idx = torch.nonzero(spatial_binary == 0).squeeze()
    
    if spatial_zero_idx.numel() == 1:
        layer_idx = spatial_zero_idx
    elif spatial_zero_idx.numel() > 1:
        layer_idx = spatial_zero_idx[-1]
    else:
        layer_idx = -1
        
    return layer_idx