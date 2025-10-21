# online_training_hook.py
import torch
import logging
from pathlib import Path
from typing import Optional
import base64
import io
from PIL import Image
import numpy as np

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmdet.apis import inference_detector
from mmdet.registry import HOOKS

from mmdet.online_training.request_queue import RequestQueue, RequestType
from mmdet.online_training.api_server import start_api_server


@HOOKS.register_module()
class OnlineTrainingAPI(Hook):
    """
    Hook to handle online training API requests.
    
    Processes inference and dataset update requests at natural
    checkpoints in the training loop.
    """
    
    priority = 'VERY_LOW'  # Run after other hooks to avoid interference
    
    def __init__(
        self,
        data_root: Optional[str] = None
    ):
        """
        Args:
            data_root: Root directory for saving images
        """
        super().__init__()
        self.request_queue = RequestQueue()
        self.api_thread = start_api_server(
            request_queue=self.request_queue,
            host="0.0.0.0",
            port=8000
        )
        self.data_root = Path(data_root) if data_root else None
        self.image_counter = 0
    
    def before_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        self._process_pending_requests(runner)
    
    def before_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        self._process_pending_requests(runner)
    
    def _process_pending_requests(self, runner: Runner):
        """Process all pending API requests."""
        if self.request_queue.is_empty():
            return
        
        requests = self.request_queue.get_all()
        if not requests:
            return
        
        print_log(
            f"\n{'='*70}\n"
            f"📨 Processing {len(requests)} API request(s) at iteration {runner.iter}\n"
            f"{'='*70}",
            logger='current',
            level=logging.INFO
        )
        
        needs_dataloader_rebuild = False
        
        for request_type, request_data, future in requests:
            try:
                if request_type == RequestType.PREDICT:
                    result = self._handle_inference(runner, request_data)
                    future.set_result(result)
                
                elif request_type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(runner, request_data)
                    future.set_result(result)
                    needs_dataloader_rebuild = True
                
            except Exception as e:
                print_log(
                    f"❌ Request failed: {e}",
                    logger='current',
                    level=logging.ERROR
                )
                future.set_exception(e)
        
        # Rebuild dataloader if dataset was modified
        if needs_dataloader_rebuild:
            self._rebuild_dataloader(runner)
        
        print_log(
            f"{'='*70}\n"
            f"✅ Requests processed, resuming training\n"
            f"{'='*70}\n",
            logger='current',
            level=logging.INFO
        )
    
    @torch.no_grad()
    def _handle_inference(self, runner: Runner, request_data: dict) -> dict:
        """Handle inference request."""
        # Decode image
        image_bytes = base64.b64decode(request_data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Run inference
        model = runner.model
        was_training = model.training
        model.eval()
        
        try:
            # with autocast(enabled=self.fp16):
            #     outputs = self.runner.model.test_step(data_batch)
            
            result = inference_detector(model, image_np)

            # Format predictions
            pred_instances = result.pred_instances
            predictions = {
                'bboxes': pred_instances.bboxes.cpu().numpy().tolist(),
                'scores': pred_instances.scores.cpu().numpy().tolist(),
                'labels': pred_instances.labels.cpu().numpy().tolist(),
            }
            
            return {
                'status': 'success',
                'predictions': predictions,
                'metadata': {
                    'iteration': runner.iter,
                    'epoch': runner.epoch,
                    'num_detections': len(predictions['bboxes'])
                }
            }
        
        finally:
            # Restore training mode
            if was_training:
                model.train()
    
    def _handle_add_sample(self, runner: Runner, request_data: dict) -> dict:
        """Handle add sample request."""
        # Get dataset
        train_loop = runner.train_loop
        dataset = train_loop.dataloader.dataset
        
        if not hasattr(dataset, 'add_sample'):
            raise RuntimeError(
                "Dataset must have 'add_sample' method. "
                "Please use OnlineTrainingDataset."
            )
        
        # Decode and save image
        image_bytes = base64.b64decode(request_data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Determine save path
        if self.data_root is None:
            self.data_root = Path(runner.work_dir) / "data"
        
        image_dir = self.data_root / "online_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        self.image_counter += 1
        filename = request_data.get('filename') or f'online_{self.image_counter:06d}.jpg'
        image_path = image_dir / filename
        
        # Save image
        image.save(image_path, 'JPEG', quality=95)
        
        # Create image info (COCO format)
        width, height = image.size
        img_info = {
            'id': self.image_counter,
            'file_name': str(image_path),
            'width': width,
            'height': height,
        }
        
        # Validate and add annotations
        annotations = request_data['annotations']
        validated_annotations = self._validate_annotations(annotations, img_info)
        
        # Add to dataset
        sample_idx = dataset.add_sample(img_info, validated_annotations)
        
        print_log(
            f"✅ Added sample {sample_idx}: {filename}, "
            f"{len(validated_annotations)} annotations, "
            f"dataset size: {len(dataset)}",
            logger='current',
            level=logging.INFO
        )
        
        return {
            'status': 'success',
            'sample_index': sample_idx,
            'image_info': img_info,
            'num_annotations': len(validated_annotations),
            'dataset_size': len(dataset),
            'metadata': {
                'iteration': runner.iter,
                'epoch': runner.epoch
            }
        }
    
    def _validate_annotations(self, annotations: list, img_info: dict) -> list:
        """Validate and normalize annotations."""
        validated = []
        
        for ann in annotations:
            if 'bbox' not in ann or 'category_id' not in ann:
                print_log(
                    f"⚠️  Skipping invalid annotation: {ann}",
                    logger='current',
                    level=logging.WARNING
                )
                continue
            
            # Validate bbox [x, y, width, height]
            bbox = ann['bbox']
            if len(bbox) != 4 or any(v < 0 for v in bbox):
                print_log(
                    f"⚠️  Invalid bbox: {bbox}",
                    logger='current',
                    level=logging.WARNING
                )
                continue
            
            x, y, w, h = bbox
            
            # Clip to image boundaries
            if x + w > img_info['width']:
                w = img_info['width'] - x
            if y + h > img_info['height']:
                h = img_info['height'] - y
            
            if w <= 0 or h <= 0:
                continue
            
            validated_ann = {
                'bbox': [x, y, w, h],
                'category_id': int(ann['category_id']),
                'area': float(w * h),
                'iscrowd': ann.get('iscrowd', 0),
            }
            
            if 'segmentation' in ann:
                validated_ann['segmentation'] = ann['segmentation']
            
            validated.append(validated_ann)
        
        return validated
    
    def _rebuild_dataloader(self, runner: Runner):
        """
        Rebuild dataloader to propagate dataset changes to worker processes.
        
        Critical: DataLoader workers have their own copy of the dataset.
        After adding samples, we must recreate the dataloader.
        """
        print_log(
            "🔄 Rebuilding dataloader to propagate dataset changes...",
            logger='current',
            level=logging.INFO
        )
        
        train_loop = runner.train_loop
        
        # Store current dataset (it has the new samples)
        current_dataset = train_loop.dataloader.dataset
        
        # Get dataloader config and update dataset
        dataloader_cfg = runner.cfg.train_dataloader.copy()
        dataloader_cfg['dataset'] = current_dataset
        
        # Build new dataloader
        new_dataloader = runner.build_dataloader(
            dataloader_cfg,
            seed=runner.seed,
        )
        
        # Replace old dataloader
        old_dataloader = train_loop.dataloader
        train_loop.dataloader = new_dataloader
        
        # Update max_iters (dataset size changed)
        train_loop._max_iters = train_loop._max_epochs * len(new_dataloader)
        
        # Clean up old dataloader (terminates old workers)
        del old_dataloader
        
        print_log(
            f"✅ Dataloader rebuilt: {len(new_dataloader)} batches per epoch",
            logger='current',
            level=logging.INFO
        )