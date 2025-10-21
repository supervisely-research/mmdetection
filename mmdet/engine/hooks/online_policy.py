from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.datasets.online_training_dataset import OnlineTrainingDataset
from mmdet.registry import HOOKS

from pycocotools.coco import COCO

class BaseOnlinePolicy(Hook):
    """Base class for online training policy hooks.

    All online training policy hooks should inherit from this class.
    """
    def __init__(self):
        self._runner: 'Runner' = None
        self._train_dataset: OnlineTrainingDataset = None

    def before_run(self, runner: 'Runner'):
        self._runner = runner
        self._train_dataset: OnlineTrainingDataset = runner.train_loop.dataloader.dataset
        assert isinstance(self._train_dataset, OnlineTrainingDataset), \
            'The training dataset must be an instance of OnlineTrainingDataset.'

    def add_sample(self, img_info: dict, annotations: list) -> int:
        return self._train_dataset.add_sample(img_info, annotations)

    @property
    def iter(self):
        return self._runner.iter


@HOOKS.register_module()
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
        ):
        self.ann_file = ann_file
        self.start_samples = start_samples
        self.add_interval = add_interval
        self.add_count = add_count
        self.coco = COCO(ann_file)
        self.all_img_ids = self.coco.getImgIds()
        print(self.all_img_ids)        
        self.current_idx = 0

    def before_run(self, runner: 'Runner'):
        super().before_run(runner)
        # Initially add start_samples samples
        for i in range(self.start_samples):
            self.add_next_sample()
    
    def add_next_sample(self):
        """Add a single new sample from the annotation file."""
        # Check if we've exhausted all samples
        if self.current_idx >= len(self.all_img_ids):
            print(f"Warning: All {len(self.all_img_ids)} samples have been added. No more samples to add.")
            return None
        
        # Get the next image ID
        img_id = self.all_img_ids[self.current_idx]
        
        # Get image info
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Add the sample to the dataset
        sample_idx = self.add_sample(img_info, annotations)
        
        # Move to next sample
        self.current_idx += 1
        
        return sample_idx
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Called after each training iteration."""
        # Check if we should add new samples at this iteration
        # Skip iteration 0, start from add_interval
        if self.iter > 0 and self.iter % self.add_interval == 0:
            print(f"\n[Iteration {self.iter}] Adding {self.add_count} new sample(s)...")
            
            added_count = 0
            for i in range(self.add_count):
                sample_idx = self.add_next_sample()
                if sample_idx is not None:
                    added_count += 1
            
            print(f"Successfully added {added_count} sample(s). Total samples: {len(self._train_dataset)}")