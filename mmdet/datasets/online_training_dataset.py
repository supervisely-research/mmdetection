# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class OnlineTrainingDataset(BaseDetDataset):
    """Dataset for incremental online training in COCO data format. This dataset must be dynamically
    added with new samples during the training process."""
    
    METAINFO: dict = dict()

    def load_data_list(self) -> List[dict]:
        return []

    def add_sample(self, img_info: dict, annontations: List[dict]) -> int:
        """Add a new sample to the dataset.

        Args:
            img_info (dict): Image information in COCO format. 'img_id' will be
                automatically generated.
            annontations (List[dict]): A list of annotations in COCO format.
        """
        # generate new img_id
        if len(self.data_list) == 0:
            new_img_id = 0
        else:
            new_img_id = max([data['img_id'] for data in self.data_list]) + 1

        img_info['img_id'] = new_img_id

        data_info = self.parse_data_info({
            'raw_img_info': img_info,
            'raw_ann_info': annontations
        })
        self.data_list.append(data_info)
        return new_img_id

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw COCO annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
