# Invariant Teacher and Equivariant Student (ITES)
This repository contains the implementation of the **AAAI2021 paper named: Invariant Teacher and Equivariant Student for Unsupervised 3D Human Pose Estimation**. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16409)

<p align="center"><img src="img/vis.gif" width="90%" alt="" /></p>

Abstract: We propose a novel method based on teacher-student learning framework for 3D human pose estimation without any 3D annotation or side information. To solve this unsupervised-learning problem, the teacher network adopts pose-dictionary-based modeling for regularization to estimate a physically plausible 3D pose. To handle the decomposition ambiguity in the teacher network, we propose a cycle-consistent architecture promoting a 3D rotation-invariant property to train the teacher network. To further improve the estimation accu- racy, the student network adopts a novel graph convolution network for flexibility to directly estimate the 3D coordinates. Another cycle-consistent architecture promoting 3D rotation-equivariant property is adopted to exploit geometry consistency, together with knowledge distillation from the teacher network to improve the pose estimation performance. We conduct extensive experiments on Human3.6M and MPI-INF-3DHP. Our method reduces the 3D joint prediction error by 11.4% compared to state-of-the-art unsupervised methods and also outperforms many weakly-supervised methods that use side information on Human3.6M.

# Requirement
* Python 3.6
* Pytorch 1.1
* Argparse
* Numpy
* Matplotlib

To have a quick installation, you can simply run:
```
pip install -r requirements.txt
```

# Data preparation
You can find the instructions for setting up the Human3.6M and results of 2D detections in [`data/README.md`](data/README.md). The code for data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). After finishing the data preparation, you should have three .npz files named data_3d_h36m.npz, data_2d_h36m_gt.npz and data_2d_h36m_cpn_ft_h36m_dbb.npz.


# Training and Testing

### Training from scratch
To train a pose-dictionary-based teacher network using ground-truth 2D detection, run
```
python train_teacher.py -k gt -e 50 -c checkpoint/teacher
```
To train a graph-convolution-based student network by a freezed teacher network using ground-truth 2D detection, run
```
python train_student.py -k gt -e 40 -c checkpoint/student --teacher_checkpoint checkpoint/teacher/ckpt_teacher.bin
```
if you want to train the student network by your own teacher network model, just modify the --teacher_checkpoint.

### Evaluating pretrained models
To evaluate a trained pose-dictionary-based teacher network using ground-truth 2D detection, run
```
python test_teacher.py -k gt --evaluate -c checkpoint/teacher/ckpt_teacher.bin
```
or using CPN 2D detection (IMG), run
```
python test_teacher.py -k cpn_ft_h36m_dbb --evaluate -c checkpoint/teacher/ckpt_teacher.bin
```

To evaluate a trained graph-convolution-based student network using ground-truth 2D detection, run
```
python test_student.py -k gt --evaluate -c checkpoint/student/ckpt_student.bin
```
or using CPN 2D detection (IMG), run
```
python test_student.py -k cpn_ft_h36m_dbb --evaluate -c checkpoint/student/ckpt_student.bin
```

### Visualization
To get visualization result of a teacher network, simply run
```
python test_teacher.py -k gt --vis -c checkpoint/teacher/ckpt_teacher.bin
```
To get visualization result of a stuent network, simply run
```
python test_student.py -k gt --vis -c checkpoint/teacher/ckpt_student.bin
```

# Acknowledgement
Thanks for the data preprocessing code provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), which is source code of the published work [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/abs/1811.11742).

We also thank for some parts of the network structure provided by the works of C3DPO ([paper](https://arxiv.org/abs/1909.02533),[code](https://github.com/facebookresearch/c3dpo_nrsfm)) and SemGCN ([paper](https://arxiv.org/abs/1904.03345),[code](https://github.com/garyzhao/SemGCN)).

# Citation
If you use this code, please cite our paper:
```
@inproceedings{xu2021invariant,
  title={Invariant Teacher and Equivariant Student for Unsupervised 3D Human Pose Estimation},
  author={Xu, Chenxin and Chen, Siheng and Li, Maosen and Zhang, Ya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3013--3021},
  year={2021}
}
```
