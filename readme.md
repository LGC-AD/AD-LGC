# Repository for LGC
Code for paper **"[Revitalizing Reconstruction Models for Multi-class Anomaly Detection via Class-Aware Contrastive Learning](1111)"**.


## 🛠️ Getting Started

### Installation
- Prepare general experimental environment
  ```shell
  pip install -r requriements.txt
  ```
  
### Dataset Preparation 
Download datasets to `data/` folder or set `self.data.root` in `configs/lgc/lgc_data.py`.
- [Real-IAD](https://realiad4ad.github.io/Real-IAD/): A new large-scale challenging industrial AD dataset, containing 30 classes with totally 151,050 images; 2,000 ∼ 5,000 resolution; 0.01% ~ 6.75% defect proportions; 1:1 ~ 1:10 defect ratio.
- [BTAD](https://github.com/pankajmishra000/VT-ADL): A real-world industrial anomaly dataset. The dataset contains a total of 2830 real-world images of 3 industrial products showcasing body and surface defects.
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/): It contains over 5000 high-resolution images divided into fifteen different object and texture categories.
- [VisA](https://github.com/amazon-science/spot-diff): It contains 12 subsets corresponding to 12 different objects as shown in the above figure. There are 10,821 images with 9,621 normal and 1,200 anomalous samples.

### Train
- Check `data` and `model` settings for the config file `configs/lgc/lgc_data.py`
- Train with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/lgc/lgc_data.py -m train`
- Train with multiple GPUs (DDP) in one node: 
  - `export nproc_per_node=8`
  - `export nnodes=1`
  - `export node_rank=0`
  - `export master_addr=YOUR_MACHINE_ADDRESS`
  - `export master_port=12315`
  - `python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c configs/lgc/lgc_data.py -m train`.
- Modify `trainer.resume_dir` to resume training. 

### Test
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Test with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/lgc/lgc_data.py -m test`
- Test with multiple GPUs (DDP) in one node:  `python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c configs/lgc/lgc_data.py -m test`.

### Visualization
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Visualize with single GPU example: `CUDA_VISIBLE_DEVICES=0 python run.py -c configs/lgc/lgc_data.py -m test vis=True vis_dir=VISUALIZATION_DIR`





### Checkpoints

- LGC with RD: [[Google Drive](https://drive.google.com/file/d/1gGDEpLWGW-MH4DP_4faDgQkQoCXdSt-S/view?usp=sharing)], [[百度云: 密码2tvt](https://pan.baidu.com/s/1MdA9AEBeyr8v7WqjifTjPA)]

---

## Acknowledgement

Our benchmark is built on [ADer](https://github.com/zhangzjn/ADer) and [RD4AD](https://github.com/hq-deng/RD4AD), thanks their extraordinary works!


---

## Citation
```angular2html

```