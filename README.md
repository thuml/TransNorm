# TransNorm
Code release for ["Transferable Normalization: Towards Improving Transferability of Deep Neural Networks"](https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks) (NeurIPS 2019)

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Training
```
Office-31

pythonn train_image.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
```
```
Office-Home

pythonn train_image.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```
```
VisDA 2017

pythonn train_image.py --gpu_id id --net ResNet50 --dset visda --test_interval 5000 --s_dset_path ../data/visda-2017/train_list.txt --t_dset_path ../data/visda-2017/validation_list.txt
```
```
Image-clef

pythonn train_image.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/b_list.txt --t_dset_path ../data/image-clef/i_list.txt
```

## Acknowledgement
This code is implemented based on the published code of CDAN and BatchNorm, and it is our pleasure to acknowledge their contributions.
CDAN (Conditional Adversarial Domain Adaptation)
BatchNorm (Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift)

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{Wang19TransNorm,
    title = {Transferable Normalization: Towards Improving Transferability of Deep Neural Networks},
    author = {Wang, Ximei and Jin, Ying and Long, Mingsheng and Wang, Jianmin and Jordan, Michael I},
    booktitle = {Advances in Neural Information Processing Systems 32},
    year = {2019}
}
```

## Contact
If you have any problem about our code, feel free to contact
- wxm17@mails.tsinghua.edu.cn
- longmingsheng@gmail.com

or describe your problem in Issues.
