# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model_teacher import Teacher_net
from common.loss import *
from common.generators_pspt import PoseGenerator,PoseGenerator_new
from common.function import *
from time import time
from common.utils import deterministic_random
import math
from torch.utils.data import DataLoader
from common.model_student import Student_net

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# bones = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16),(7,8),(8,9),(9,10)]
# avg_bone_length = [0.13452314, 0.4557758 , 0.4516843 , 0.13452327, 0.45577532 ,0.4516843,0.2456195 ,0.15755513, 0.28403646,
#                  0.24994996 ,0.15755513, 0.28403646, 0.24994996 ,0.2543286 , 0.11673375 ,0.11501251]
# bone_pairs = [(7, 11, 7, 14), (11, 12, 14, 15), (12, 13, 15, 16), (0, 4, 0, 1), (4, 5, 1, 2), (5, 6, 2, 3)]

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset

    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        if args.dataset != 'gt' and action =='Directions' and subject =='S11':
            continue
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                                        subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            #kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            kps -= kps[:,:1]
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(np.tile((cam['intrinsic'])[None,:],(len(poses_2d[i]),1)))

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                out_camera_params[i] = out_camera_params[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

model_pos_train = Teacher_net(poses_valid_2d[0].shape[-2],dataset.skeleton().num_joints(),poses_valid_2d[0].shape[-1])
model_pos = Teacher_net(poses_valid_2d[0].shape[-2],dataset.skeleton().num_joints(),poses_valid_2d[0].shape[-1])

adj = adj_mx_from_skeleton(dataset.skeleton())
model_depth_train = Student_net(adj, args.hid_dim, num_layers=args.n_blocks, p_dropout=0.0,
                       nodes_group=dataset.skeleton().joints_group())
model_depth = Student_net(adj, args.hid_dim, num_layers=args.n_blocks, p_dropout=0.0,
                       nodes_group=dataset.skeleton().joints_group())

model_params = 0
for parameter in model_depth.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    model_depth_train = model_depth_train.cuda()
    model_depth = model_depth.cuda()

chk_filename = args.teacher_checkpoint
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model_pos_train.load_state_dict(checkpoint['model_pos'],strict=False)
model_pos.load_state_dict(checkpoint['model_pos'],strict=False)

valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, cameras_valid),
                                      batch_size=1024, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)

cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

lr = args.learning_rate

optimizer = torch.optim.SGD(model_depth_train.parameters(), lr=lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = optim.Adam(model_depth_train.parameters(), lr=lr, amsgrad=True)

lr_decay = args.lr_decay

losses_3d_train_rp = []
losses_3d_train_cs = []
losses_3d_valid_p1 = []
losses_3d_valid_p2 = []

epoch = 0
initial_momentum = 0.1
final_momentum = 0.001

train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, cameras_train), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)

print('*** Generating depth labels... ***')
model_pos.load_state_dict(model_pos_train.state_dict())
model_pos.eval()
pose_2d_ft = []
pose_3d_ft = []
with torch.no_grad():
    for i, (inputs_3d, inputs_2d, inputs_scale) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()

        # Predict 3D poses
        preds = model_pos(inputs_2d)
        shape_camera_coord = preds['shape_camera_coord']
        pose_2d_ft.append(inputs_2d.cpu())
        pose_3d_ft.append(shape_camera_coord[:,:,2:3].cpu())

train_loader_new = DataLoader(PoseGenerator_new(pose_3d_ft, pose_2d_ft), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)

print('*** Start teaching... ***')
while epoch < args.epochs:
    start_time = time()
    epoch_loss_3d_train_rp = 0
    epoch_loss_3d_train_cs = 0
    N = 0
    model_depth_train.train()
    # Regular supervised scenario
    for i, (inputs_depth, inputs_2d) in enumerate(train_loader_new):
        if torch.cuda.is_available():
            inputs_depth = inputs_depth.cuda()
            inputs_2d = inputs_2d.cuda()

        optimizer.zero_grad()
        # Predict 3D poses
        preds = model_depth_train(inputs_2d)
        loss_reprojection = mpjpe(preds['depth'], inputs_depth)
        loss_consistancy = preds['l_reconstruct']
        loss_total = args.reploss_weight * loss_reprojection + loss_consistancy 
        epoch_loss_3d_train_rp += inputs_2d.shape[0] * loss_reprojection.item()
        epoch_loss_3d_train_cs += inputs_2d.shape[0] * loss_total.item()
        N += inputs_2d.shape[0]

        loss_total.backward()

        optimizer.step()

    losses_3d_train_rp.append(epoch_loss_3d_train_rp / N)
    losses_3d_train_cs.append(epoch_loss_3d_train_cs / N)

    # End-of-epoch evaluation
    with torch.no_grad():
        model_depth.load_state_dict(model_depth_train.state_dict())
        model_depth.eval()

        epoch_error_p1 = 0
        epoch_error_p2 = 0
        N = 0

        if not args.no_eval:
            # Evaluate on test set
            for i, (inputs_3d, inputs_2d, inputs_scale) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()

                # Predict 3D poses
                #inputs_2d=inputs_3d[:,:,:2]
                preds = model_depth(inputs_2d)
                shape_camera_coord = preds['shape_3d']
                shape_camera_coord_flip = shape_camera_coord.clone()
                shape_camera_coord_flip[:,:,2] = -shape_camera_coord[:,:,2]
                shape_camera_coord = calibrate_by_scale(shape_camera_coord,inputs_3d)
                shape_camera_coord_flip = calibrate_by_scale(shape_camera_coord_flip,inputs_3d)

                shape_camera_coord = shape_camera_coord - shape_camera_coord[:,0:1,:]
                shape_camera_coord_flip = shape_camera_coord_flip - shape_camera_coord_flip[:,0:1,:]
                inputs_3d = inputs_3d - inputs_3d[:,0:1,:]
                inputs_scale = np.asarray(inputs_scale)

                dist = calc_dist(shape_camera_coord, inputs_3d)
                p_dist = p_mpjpe(shape_camera_coord, inputs_3d)
                dist_flip = calc_dist(shape_camera_coord_flip, inputs_3d)
                p_dist_flip = p_mpjpe(shape_camera_coord_flip,inputs_3d)

                dist_best = np.minimum(dist,dist_flip)
                p_dist_best = np.minimum(p_dist,p_dist_flip)

                dist_best = dist_best * inputs_scale
                p_dist_best = p_dist_best * inputs_scale
                loss_3d_p1 = dist_best.mean()
                loss_3d_p2 = p_dist_best.mean()
                epoch_error_p1 += inputs_3d.shape[0] * loss_3d_p1
                epoch_error_p2 += inputs_3d.shape[0] * loss_3d_p2
                N += inputs_3d.shape[0]

            losses_3d_valid_p1.append(epoch_error_p1 / N)
            losses_3d_valid_p2.append(epoch_error_p2 / N)

    elapsed = (time() - start_time) / 60

    if args.no_eval:
        print('[%d] time %.2f lr %f reprojection_loss %f consistent loss %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train_rp[-1] * 1000,
            losses_3d_train_cs[-1] * 1000,))
    else:
        print('[%d] time %.2f lr %f reprojection_loss %f consistent loss %f MPJPE %f P-MPJPE %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train_rp[-1] * 1000,
            losses_3d_train_cs[-1] * 1000,
            losses_3d_valid_p1[-1] * 1000,
            losses_3d_valid_p2[-1] * 1000))

    # Decay learning rate exponentially
    if (epoch+1) % args.epoch_lr_decay == 0:
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
    epoch += 1

    # if (epoch+1) == 15 or (epoch+1) == 25:
    #     lr *= 0.1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1

    # Save checkpoint if necessary
    if epoch >= 1:
        # chk_path = os.path.join('checkpoint/model_depth_mygcn', 'epoch2_{}.bin'.format(epoch))
        chk_path = os.path.join(args.checkpoint, 'stu_model_epoch_{}.bin'.format(epoch))
        print('Saving checkpoint to', chk_path)

        torch.save({
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model_depth_train.state_dict(),
        }, chk_path)
