Build mmcv.
I mananged to build mmcv 2.2.0 with cuda 12.8.

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
# pip install -r requirements/optional.txt  # install ninja to speed up compilation, but didn't work for me
# export CUDA_HOME=/usr/local/cuda  # not neccessary?
pip install -e . -v
python .dev_scripts/check_installation.py  # check if mmcv is installed correctly
```

```bash
mkdir weights
cd weights
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
pip install -r requirements/multimodal.txt
ln -s /root/volume/data data
```

change mmdet/__init__.py line 17 to

```python
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
```

/usr/local/lib/python3.10/dist-packages/mmengine/runner/checkpoint.py line 347 to

```python
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
```
