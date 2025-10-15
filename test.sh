WORK_DIR="work_dirs/swin-l_finetune-bs2-lr2e-5-decay-ema-bert-on-300e-v2"

python tools/test.py \
    "$WORK_DIR/swin-l_finetune.py" \
    "$WORK_DIR/best_coco_bbox_mAP_epoch_100.pth" \
    --work-dir "$WORK_DIR/test"
# --show-dir vis_results