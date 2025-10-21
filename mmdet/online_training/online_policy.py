from mmdet.registry import ONLINE_POLICY
from mmengine.runner import Runner
from mmdet.datasets.online_training_dataset import OnlineTrainingDataset

# online_policy = dict(
#     type='SimplePolicy',
#     ann_file=data_root + train_ann_file_30shot,
#     start_samples=5,  # start with 5 samples
#     add_interval=10,  # add new samples every 10 steps
#     add_count=1,    # add 1 new sample each time
# )

class BaseOnlinePolicy:
    """Base class for online training policy.

    All online training policies should inherit from this class.
    """

    def __init__(self, runner: 'Runner'):
        self._runner = runner
        self._train_dataset: OnlineTrainingDataset = runner.train_dataloader.dataset
        assert isinstance(self._train_dataset, OnlineTrainingDataset), \
            'The training dataset must be an instance of OnlineTrainingDataset.'

    def add_sample(self, img_info: dict, annotations: list) -> int:
        return self._train_dataset.add_sample(img_info, annotations)


@ONLINE_POLICY.register_module()
class SimplePolicy(BaseOnlinePolicy):
    """A simple online policy for baseline.

    This policy adds new samples from a predefined annotation file at fixed
    intervals during training.
    """

    def __init__(
            self,
            ann_file: str,
            start_samples: int,
            add_interval: int,
            add_count: int,
            runner: 'Runner'
        ):
        super().__init__(runner)
        self.ann_file = ann_file
        self.start_samples = start_samples
        self.add_interval = add_interval
        self.add_count = add_count