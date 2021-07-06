import numpy as np
import torch
from torch.utils.data import Dataset
import random


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, cam):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._cam = np.concatenate(cam)
        self.focal_length = self._cam[:,:2]

        self._poses_2d = self._poses_2d / np.tile(self.focal_length[:,None,:],(1,self._poses_2d.shape[1],1))
        self._poses_2d = self.normalize_2d(self._poses_2d)

        self._poses_3d = self._poses_3d - self._poses_3d[:,0:1,:]
        self._scale = np.ones(((len(self._poses_3d),1)))

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...'.format(self._poses_2d.shape[0]))

    def normalize_2d(self,pose):
        # pose:(N,J,2)
        mean_bone = np.mean(np.linalg.norm(pose[:,0:1,:]-pose[:,10:11,:],axis=2,ord=2)) #hip to head
        c = 5
        scale = (1/c) / mean_bone
        pose = pose * scale
        return pose

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_scale = self._scale[index]
        # out_tz = self._tz[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        return out_pose_3d, out_pose_2d,out_scale

    def __len__(self):
        return len(self._poses_2d)

class PoseGenerator_new(Dataset):
    def __init__(self, poses_3d, poses_2d):
        assert poses_3d is not None

        self._poses_3d = torch.cat(poses_3d,dim=0)
        self._poses_2d = torch.cat(poses_2d,dim=0)

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._poses_2d)