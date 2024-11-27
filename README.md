# pointrcnn-mdn

A Mixture Density Networks (MDN) implementation of pointrcnn, based on [openpcdet](https://github.com/open-mmlab/OpenPCDet)

To understand why MDN is used for uncertainty evaluation, refer to [this paper](https://arxiv.org/pdf/1709.02249).

## Usage

train with:

```sh
# car only detection model
python train.py --cfg_file cfgs/kitti_models/pointrcnn_mdn_car.yaml 
```

##  Examples

TODO
