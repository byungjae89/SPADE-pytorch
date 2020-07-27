# Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE) in PyTorch

PyTorch implementation of [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/abs/2005.02357) (SPADE).  

**SPADE** presents an anomaly segmentation approach which does not require a training stage.  
It is fast, robust and achieves SOTA on `MVTec AD` dataset.  

* *We used K=5 nearest neighbors, which differs from the original paper K=50.*


## Prerequisites
* python 3.6+
* PyTorch 1.5+
* sklearn, matplotlib

Install prerequisites with:  
```
pip install -r requirements.txt
```

If you already download [`MVTec AD`](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset, move a file to `data/mvtec_anomaly_detection.tar.xz`.  
If you don't have a dataset file, it will be automatically downloaded during the code running.

## Usage

To test **SPADE** on `MVTec AD` dataset:
```
cd src
python main.py
```

After running the code above, you can see the ROCAUC results in `src/result/roc_curve.png`

## Results

Below is the implementation result of the test set ROCAUC on the `MVTec AD` dataset.  

### 1. Image-level anomaly detection accuracy (ROCAUC %)

| | Paper | Implementation |
| - | - | - |
| bottle | - | 97.2 |
| cable | - | 84.8 |
| capsule | - | 89.7 |
| carpet | - | 92.8 |
| grid | - | 47.3 |
| hazelnut | - | 88.1 |
| leather | - | 95.4 |
| metal_nut | - | 71.0 |
| pill | - | 80.1 |
| screw | - | 66.7 |
| tile | - | 96.5 |
| toothbrush | - | 88.9 |
| transistor | - | 90.3 |
| wood | - | 95.8 |
| zipper | - | 96.6 |
| Average | 85.5 | 85.4 |

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper | Implementation |
| - | - | - |
| bottle | 98.4 | 97.0 |
| cable | 97.2 | 92.3 |
| capsule | 99.0 | 98.4 |
| carpet | 97.5 | 98.9 |
| grid | 93.7 | 98.3 |
| hazelnut | 99.1 | 98.5 |
| leather | 97.6 | 99.3 |
| metal_nut | 98.1 | 97.1 |
| pill | 96.5 | 95.0 |
| screw | 98.9 | 99.1 |
| tile | 87.4 | 92.8 |
| toothbrush | 97.9 | 98.8 |
| transistor | 94.1 | 86.6 |
| wood | 88.5 | 95.3 |
| zipper | 96.5 | 98.6 |
| Average | 96.5 | 96.4 |

### ROC Curve 

![roc](./assets/roc_curve.png)

### Localization results  

![bottle](./assets/bottle_000.png)  
![cable](./assets/cable_000.png)  
![capsule](./assets/capsule_000.png)  
![carpet](./assets/carpet_000.png)  
![grid](./assets/grid_000.png)  
![hazelnut](./assets/hazelnut_000.png)  
![leather](./assets/leather_000.png)  
![metal_nut](./assets/metal_nut_000.png)  
![pill](./assets/pill_000.png)  
![screw](./assets/screw_000.png)  
![tile](./assets/tile_000.png)  
![toothbrush](./assets/toothbrush_000.png)  
![transistor](./assets/transistor_000.png)  
![wood](./assets/wood_000.png)  
![zipper](./assets/zipper_000.png)  














