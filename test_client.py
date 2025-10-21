# test_client.py
"""
Test client for Online Training API.

This script tests the online training functionality by:
1. Waiting for API server to be ready
2. Adding initial 3 samples
3. Continuously adding samples every 10 seconds
4. Making inference requests every 30 seconds (starting after 20s)
"""

import requests
import base64
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import threading
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class OnlineTrainingTestClient:
    """Test client for online training API."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        data_root: str = 'data/insulator-defect-detection/',
        train_ann_file: str = 'fsod_coco_idx0/train_30shot_seed0.json',
        train_dir: str = 'project/train/img',
        val_dir: str = 'project/val/img',
    ):
        self.api_url = api_url
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / train_dir
        self.val_dir = self.data_root / val_dir
        
        # Load COCO annotations
        ann_path = self.data_root / train_ann_file
        logger.info(f"Loading annotations from: {ann_path}")
        
        with open(ann_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Create image_id to annotations mapping
        self.img_to_anns = self._group_annotations_by_image()
        
        # Track which samples have been added
        self.sample_index = 0
        self.total_samples = len(self.images)
        
        logger.info(f"Loaded {self.total_samples} images with {len(self.annotations)} annotations")
    
    def _group_annotations_by_image(self) -> Dict[int, List[Dict]]:
        """Group annotations by image_id."""
        img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        return img_to_anns
    
    def wait_for_server(self, timeout: int = 60, check_interval: float = 1.0):
        """Wait for API server to be ready."""
        logger.info(f"Waiting for API server at {self.api_url}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ API server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(check_interval)
        
        logger.error(f"❌ API server did not become ready within {timeout}s")
        return False
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def get_sample(self, index: int) -> tuple:
        """
        Get a sample (image + annotations) by index.
        
        Returns:
            tuple: (image_base64, annotations, image_info)
        """
        if index >= self.total_samples:
            logger.warning(f"Sample index {index} out of range ({self.total_samples} total)")
            return None, None, None
        
        # Get image info
        img_info = self.images[index]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # Construct full image path
        image_path = self.train_dir / img_filename
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None, None, None
        
        # Encode image
        image_b64 = self.encode_image(image_path)
        
        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])
        
        # Format annotations (keep only necessary fields)
        formatted_anns = []
        for ann in annotations:
            formatted_anns.append({
                'bbox': ann['bbox'],  # [x, y, width, height]
                'category_id': ann['category_id'],
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0),
            })
        
        return image_b64, formatted_anns, img_info
    
    def add_sample(self, index: int) -> Dict[str, Any]:
        """Add a sample to the training dataset."""
        image_b64, annotations, img_info = self.get_sample(index)
        
        if image_b64 is None:
            return None
        
        logger.info(f"Adding sample {index + 1}/{self.total_samples}: {img_info['file_name']} "
                   f"with {len(annotations)} annotations")
        
        # Prepare request
        payload = {
            'image': image_b64,
            'annotations': annotations,
            'filename': Path(img_info['file_name']).name
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/add_sample",
                json=payload,
                timeout=60  # Allow time for training to pause
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"✅ Sample added successfully: dataset_size={result['dataset_size']}, "
                       f"iteration={result['metadata']['iteration']}")
            
            return result
        
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout while adding sample {index}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error adding sample {index}: {e}")
            return None
    
    def predict(self, image_index: int = None) -> Dict[str, Any]:
        """Run inference on an image."""
        # If no index specified, use a validation image or random sample
        if image_index is None:
            image_index = 0  # Use first sample for inference
        
        image_b64, _, img_info = self.get_sample(image_index)
        
        if image_b64 is None:
            return None
        
        logger.info(f"Running inference on: {img_info['file_name']}")
        
        payload = {'image': image_b64}
        
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            num_detections = result['metadata']['num_detections']
            iteration = result['metadata']['iteration']
            
            logger.info(f"✅ Inference successful: {num_detections} detections found "
                       f"at iteration {iteration}")
            
            # Log some detection details
            if num_detections > 0:
                preds = result['predictions']
                for i in range(min(3, num_detections)):  # Show first 3
                    logger.info(f"   Detection {i+1}: "
                              f"label={preds['labels'][i]}, "
                              f"score={preds['scores'][i]:.3f}, "
                              f"bbox={preds['bboxes'][i]}")
            
            return result
        
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout during inference")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error during inference: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get API status."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return None
    
    def populate_initial_samples(self, count: int = 3):
        """Add initial samples to the dataset."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📥 POPULATING INITIAL {count} SAMPLES")
        logger.info(f"{'='*70}\n")
        
        for i in range(count):
            if self.sample_index >= self.total_samples:
                logger.warning("No more samples available")
                break
            
            self.add_sample(self.sample_index)
            self.sample_index += 1
            
            # Small delay between samples
            if i < count - 1:
                time.sleep(2)
        
        logger.info(f"\n✅ Initial population complete: {count} samples added\n")
    
    def continuous_add_samples(self, interval: int = 10):
        """Continuously add samples at specified interval."""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 STARTING CONTINUOUS SAMPLE ADDITION (every {interval}s)")
        logger.info(f"{'='*70}\n")
        
        while True:
            time.sleep(interval)
            
            if self.sample_index >= self.total_samples:
                logger.info("All samples have been added. Stopping continuous addition.")
                break
            
            logger.info(f"\n--- Adding sample (every {interval}s) ---")
            self.add_sample(self.sample_index)
            self.sample_index += 1
    
    def continuous_inference(self, interval: int = 30, initial_delay: int = 20):
        """Continuously run inference at specified interval."""
        # Wait for initial delay
        logger.info(f"\n{'='*70}")
        logger.info(f"⏳ WAITING {initial_delay}s BEFORE STARTING INFERENCE")
        logger.info(f"{'='*70}\n")
        
        time.sleep(initial_delay)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 STARTING CONTINUOUS INFERENCE (every {interval}s)")
        logger.info(f"{'='*70}\n")
        
        inference_count = 0
        
        while True:
            inference_count += 1
            
            logger.info(f"\n--- Inference #{inference_count} (every {interval}s) ---")
            
            # Cycle through first few images for inference
            image_idx = (inference_count - 1) % min(5, self.total_samples)
            self.predict(image_idx)
            
            # Check status
            status = self.get_status()
            if status:
                logger.info(f"📊 Status: iteration={status.get('iteration', 'N/A')}, "
                          f"dataset_size={status.get('dataset_size', 'N/A')}")
            
            time.sleep(interval)
    
    def run_test(self):
        """Run the complete test scenario."""
        logger.info("\n" + "="*70)
        logger.info("🚀 ONLINE TRAINING API TEST CLIENT")
        logger.info("="*70 + "\n")
        
        # Step 1: Wait for server
        if not self.wait_for_server(timeout=60):
            logger.error("Server not available. Exiting.")
            return
        
        time.sleep(2)
        
        # Step 2: Populate initial samples
        self.populate_initial_samples(count=3)
        
        time.sleep(2)
        
        # Step 3 & 4 & 5: Start concurrent operations
        # Thread 1: Add samples every 10 seconds
        add_thread = threading.Thread(
            target=self.continuous_add_samples,
            args=(10,),  # Every 10 seconds
            daemon=True,
            name="AddSampleThread"
        )
        
        # Thread 2: Inference every 30 seconds (starting after 20s)
        inference_thread = threading.Thread(
            target=self.continuous_inference,
            args=(30, 20),  # Every 30s, start after 20s
            daemon=True,
            name="InferenceThread"
        )
        
        # Start threads
        add_thread.start()
        inference_thread.start()
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ Test threads started!")
        logger.info("   - Adding samples every 10 seconds")
        logger.info("   - Inference every 30 seconds (starting after 20s)")
        logger.info(f"{'='*70}\n")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
                logger.info("\n--- Periodic Status Check ---")
                status = self.get_status()
                if status:
                    logger.info(f"Current status: {status}")
        
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Test interrupted by user. Exiting...")


def main():
    """Main entry point."""
    # Configuration
    API_URL = "http://localhost:8000"
    DATA_ROOT = 'data/insulator-defect-detection/'
    TRAIN_ANN_FILE = 'fsod_coco_idx0/train_30shot_seed0.json'
    TRAIN_DIR = 'project/train/img'
    VAL_DIR = 'project/val/img'
    
    # Create client
    client = OnlineTrainingTestClient(
        api_url=API_URL,
        data_root=DATA_ROOT,
        train_ann_file=TRAIN_ANN_FILE,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR
    )
    
    # Run test
    client.run_test()


if __name__ == '__main__':
    main()