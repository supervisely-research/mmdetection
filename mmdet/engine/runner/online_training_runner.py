from mmdet.registry import RUNNERS

@RUNNERS.register_module()
class OnlineTrainingRunner:
    """Runner for online training.

    This runner does not load dataset at the beginning of training.
    New samples will be added to the dataset during the training process.

    The online policy is configurable with `online_cfg` in the config file."""

    