import torch.nn.functional as F
import torch
import numpy as np
import math
import cv2
import scipy.sparse as sp


def so3_exponential_map(log_rot, eps: float = 0.0001):
    """
    Convert a batch of logarithmic representations of rotation matrices
    `log_rot` to a batch of 3x3 rotation matrices using Rodrigues formula.
    The conversion has a singularity around 0 which is handled by clamping
    controlled with the `eps` argument.
    Args:
        log_rot: batch of vectors of shape `(minibatch , 3)`
        eps: a float constant handling the conversion singularity around 0
    Returns:
        batch of rotation matrices of shape `(minibatch , 3 , 3)`
    Raises:
        ValueError if `log_rot` is of incorrect shape
    """

    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError('Input tensor shape has to be Nx3.')

    nrms = (log_rot * log_rot).sum(1)
    phis = torch.clamp(nrms, 0.).sqrt()
    phisi = 1. / (phis + eps)
    fac1 = phisi * phis.sin()
    fac2 = phisi * phisi * (1. - phis.cos())
    ss = hat(log_rot)

    R = fac1[:, None, None] * ss + \
        fac2[:, None, None] * torch.bmm(ss, ss) + \
        torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]

    return R

def hat(v: torch.Tensor):
    """
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: batch of vectors of shape `(minibatch , 3)`
    Returns:
        batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`
    Raises:
        ValueError if `v` is of incorrect shape
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError('Input vectors have to be 3-dimensional.')

    h = v.new_zeros(N, 3, 3)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def rand_rot(N, dtype=None, max_rot_angle=float(math.pi), axes=(1, 1, 1), get_ss=False):

    rand_axis = torch.zeros((N, 3)).type(dtype).normal_()

    # apply the axes mask
    axes = torch.Tensor(axes).type(dtype)
    rand_axis = axes[None, :] * rand_axis

    rand_axis = F.normalize(rand_axis, dim=1, p=2)
    rand_angle = torch.ones(N).type(dtype).uniform_(0, max_rot_angle)
    R_ss_rand = rand_axis * rand_angle[:, None]
    R_rand = so3_exponential_map(R_ss_rand)

    if get_ss:
        return R_rand, R_ss_rand
    else:
        return R_rand

def rand_rot_and_inverse(N, dtype=None, max_rot_angle=float(math.pi), axes=(1, 1, 1), get_ss=False):

    rand_axis = torch.zeros((N, 3)).type(dtype).normal_()

    # apply the axes mask
    axes = torch.Tensor(axes).type(dtype)
    rand_axis = axes[None, :] * rand_axis

    rand_axis = F.normalize(rand_axis, dim=1, p=2)
    rand_angle = torch.ones(N).type(dtype).uniform_(0, max_rot_angle)
    rand_angle_inverse = 2 * float(math.pi) - rand_angle
    R_ss_rand = rand_axis * rand_angle[:, None]
    R_ss_rand_inverse = rand_axis * rand_angle_inverse[:, None]
    R_rand = so3_exponential_map(R_ss_rand)
    R_rand_inverse = so3_exponential_map(R_ss_rand_inverse)

    if get_ss:
        return R_rand, R_ss_rand, R_rand_inverse
    else:
        return R_rand, R_rand_inverse

def split_seq(pose_3d_ft,pose_2d_ft,num_list):
    assert len(pose_3d_ft) == len(pose_2d_ft)
    assert np.sum(num_list) == len(pose_2d_ft),print(np.sum(num_list),len(pose_2d_ft))
    split_list = np.cumsum(num_list)
    pose_2d_seq = []
    pose_3d_seq = []
    for i in range(len(split_list)):
        if i == 0:
            start_idx = 0
            end_idx = split_list[i]
        else:
            start_idx = split_list[i-1]
            end_idx = split_list[i]
        pose_2d_seq.append(pose_2d_ft[start_idx:end_idx])
        pose_3d_seq.append(pose_3d_ft[start_idx:end_idx])
    return pose_3d_seq, pose_2d_seq

def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)

def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx