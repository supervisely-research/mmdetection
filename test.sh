WORK_DIR="work_dirs/swin-l_finetune-bs2-lr5e-5-bert-on"

python tools/test.py \
    "$WORK_DIR/swin-l_finetune.py" \
    "$WORK_DIR/best_coco_bbox_mAP_epoch_50.pth" \
    --work-dir "$WORK_DIR/test" \
    --show-dir "vis_results" \
    --cfg-options "test_dataloader.dataset.ann_file=fsod_coco_idx0/val_50.json" "test_evaluator.ann_file=data/insulator-defect-detection/fsod_coco_idx0/val_50.json"
