import torch
import torch.nn as nn
from common.function import so3_exponential_map
from common.camera import *
from common.function import *
from common.loss import *

class Teacher_net(nn.Module):
    def __init__(self, num_joints_in, num_joints_out, in_features=2, n_fully_connected=1024, n_layers=6, dict_basis_size=12, weight_init_std = 0.01):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.n_fully_connected = n_fully_connected
        self.n_layers = n_layers
        self.dict_basis_size = dict_basis_size

        self.fe_net = nn.Sequential(
            *self.make_trunk(dim_in=self.num_joints_in * 2,
                             n_fully_connected=self.n_fully_connected,
                             n_layers=self.n_layers))
        
        self.alpha_layer = conv1x1(self.n_fully_connected,
                            self.dict_basis_size,
                            std=weight_init_std)
        self.shape_layer = conv1x1(self.dict_basis_size, 3 * num_joints_in,
                                   std=weight_init_std)
        self.rot_layer = conv1x1(self.n_fully_connected, 3,
                                 std=weight_init_std)
        self.trans_layer = conv1x1(self.n_fully_connected, 1,
                                 std=weight_init_std)
        self.cycle_consistent = True
        self.z_augment = False
        self.z_augment_angle = 0.2

    def make_trunk(self,
                   n_fully_connected=None,
                   dim_in=None,
                   n_layers=None,
                   use_bn=True):

        layer1 = ConvBNLayer(dim_in,
                             n_fully_connected,
                             use_bn=use_bn)
        layers = [layer1]

        for l in range(n_layers):
            layers.append(ResLayer(n_fully_connected,
                                   int(n_fully_connected/4)))

        return layers

    def forward(self, input_2d, align_to_root=False):
        assert input_2d.shape[1] == self.num_joints_in
        assert input_2d.shape[2] == self.in_features

        preds = {}
        ba = input_2d.shape[0]
        dtype = input_2d.type()
        input_2d_norm, root = self.normalize_keypoints(input_2d)
        if self.z_augment:
            R_rand = rand_rot(ba,
                              dtype=dtype,
                              max_rot_angle=float(self.z_augment_angle),
                              axes=(0, 0, 1))
            input_2d_norm = torch.matmul(input_2d_norm,R_rand[:,0:2,0:2])
        preds['keypoints_2d'] = input_2d_norm
        preds['kp_mean'] = root
        input_flatten = input_2d_norm.view(-1,self.num_joints_in*2)
        feats = self.fe_net(input_flatten[:,:, None, None])
        
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        shape_invariant = self.shape_layer(shape_coeff[:, :, None, None])[:, :, 0, 0]
        shape_invariant = shape_invariant.view(ba, self.num_joints_out, 3)

        R_log = self.rot_layer(feats)[:,:,0,0]
        R = so3_exponential_map(R_log)
        T = R_log.new_zeros(ba, 3)       # no global depth offset

        scale = R_log.new_ones(ba)
        shape_camera_coord = self.rotate_and_translate(shape_invariant, R, T, scale)
        shape_image_coord = shape_camera_coord[:,:,0:2]/torch.clamp(5 + shape_camera_coord[:,:,2:3],min=1) # Perspective projection

        preds['camera'] = R
        preds['shape_camera_coord'] = shape_camera_coord
        preds['shape_coeff'] = shape_coeff
        preds['shape_invariant'] = shape_invariant
        preds['l_reprojection'] = mpjpe(shape_image_coord,input_2d_norm)
        preds['align'] = align_to_root

        if self.cycle_consistent:
            preds['l_cycle_consistent'] = self.cycle_consistent_loss(preds)
        return preds

    def cycle_consistent_loss(self, preds, class_mask=None):

        shape_invariant = preds['shape_invariant']
        if preds['align']:
            shape_invariant_root = shape_invariant - shape_invariant[:,0:1,:]
        else:
            shape_invariant_root = shape_invariant
        dtype = shape_invariant.type()
        ba = shape_invariant.shape[0]

        n_sample = 4
        # rotate the canonical point
        # generate random rotation around all axes
        R_rand = rand_rot(ba * n_sample,
                dtype=dtype,
                max_rot_angle=3.1415926,
                axes=(1, 1, 1))

        unrotated = shape_invariant_root.view(-1,self.num_joints_out,3).repeat(n_sample, 1, 1)
        rotated = torch.bmm(unrotated,R_rand)
        rotated_2d = rotated[:,:,0:2] / torch.clamp(5 + rotated[:,:,2:3],min=1)

        repred_result = self.reconstruct(rotated_2d)  

        a, b = repred_result['shape_invariant'], unrotated

        l_cycle_consistent = mpjpe(a,b)

        return l_cycle_consistent

    def reconstruct(self, rotated_2d):
        preds = {}

        # batch size
        ba = rotated_2d.shape[0]
        # reshape and pass to the network ...
        l1_input = rotated_2d.contiguous().view(ba, 2 * self.num_joints_in)

        # pass to network
        feats = self.fe_net(l1_input[:, :, None, None])
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        shape_pred = self.shape_layer(
                shape_coeff[:, :, None, None])[:, :, 0, 0]

        shape_pred = shape_pred.view(ba,self.num_joints_out,3)
        preds['shape_coeff'] = shape_coeff
        preds['shape_invariant'] = shape_pred

        return preds

    def rotate_and_translate(self, S, R, T, s):
        out = torch.bmm(S, R) + T[:,None,:]
        return out

    def normalize_keypoints(self,
                            kp_loc,
                            rescale=1.):
        # center around the root joint
        kp_mean = kp_loc[:, 0, :]
        kp_loc_norm = kp_loc - kp_mean[:, None, :]
        kp_loc_norm = kp_loc_norm * rescale

        return kp_loc_norm, kp_mean

    def normalize_3d(self,kp):
        ls = torch.norm(kp[:,1:,:],dim=2)
        scale = torch.mean(ls,dim=1)
        kp = kp / scale[:,None,None] * 0.5
        return kp

def pytorch_ge12():
    v = torch.__version__
    v = float('.'.join(v.split('.')[0:2]))
    return v >= 1.2

def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv

class ConvBNLayer(nn.Module):
    def __init__(self, inplanes, planes, use_bn=True, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv1x1(inplanes, planes)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            if pytorch_ge12():
                self.bn1.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        return out

class ResLayer(nn.Module):
    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn1.weight.data.uniform_(0., 1.)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn2.weight.data.uniform_(0., 1.)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if pytorch_ge12():
            self.bn3.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.skip = inplanes == (planes*self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            out += residual
        out = self.relu(out)

        return out
