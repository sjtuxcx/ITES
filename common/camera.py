import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse
import math


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def normalize_coordinates(X, w=1000, h=1000):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - torch.Tensor([1., h / w]).cuda()

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def camera_projection(shape,projection_type='orthographic'):
    depth = shape[:, :, 2:3]
    if projection_type == 'perspective':
        perspective_depth_threshold = 0.1
        depth = torch.clamp(depth, perspective_depth_threshold)
        projections = shape[:, :, 0:2] / depth
    elif projection_type == 'orthographic':
        projections = shape[:, :,0:2]
    else:
        raise ValueError('no such projection type %s' % self.projection_type)

    return projections, depth

def camera_unprojection(kp_loc, depth, rescale=float(1),projection_type='orthographic'):
    depth = depth / rescale
    if projection_type == 'perspective':
        shape = torch.cat((kp_loc * depth, depth), dim=2)
    elif projection_type == 'orthographic':
        shape = torch.cat((kp_loc, depth), dim=2)
    else:
        raise ValueError('no such projection type %s' %
                         self.projection_type)
    return shape

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c


def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
      P: Nx3 points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: (scalar) Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
    Returns
      Proj: Nx2 points in pixel space
      D: 1xN depth of each point in camera space
      radial: 1xN radial distortion per point
      tan: 1xN tangential distortion per point
      r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


def pinv(A):
    """
    Return the pseudoinverse of A,
    without invoking the SVD in torch.pinverse().

    Could also use (but doesn't avoid the SVD):
        R.pinverse().matmul(Q.t())
    """
    rows,cols = A.shape
    if rows >= cols:
        Q,R = torch.qr(A)
        return R.inverse().mm(Q.t())
    else:
        Q,R = torch.qr(A.t())
        return R.inverse().mm(Q.t()).t()

def calibrate_by_procrustes(points3d, camera, gt):
    """Calibrates the predictied 3d points by Procrustes algorithm.
    This function estimate an orthonormal matrix for aligning the predicted 3d
    points to the ground truth. This orhtonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.
    """
    # Shift the center of points3d to the origin
    points3d = np.asarray(points3d.cpu())
    if camera is not None:
        camera = np.asarray(camera.cpu())
    gt = np.asarray(gt.cpu())
    if camera is not None:
        singular_value = np.linalg.norm(camera, 2)
        camera = camera / singular_value
        points3d = points3d * singular_value
    scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
    points3d = points3d * scale
    U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
    rot = U.dot(Vh)
    out_3d = points3d.dot(rot)
    if camera is not None:
        out_camera = rot.T.dot(camera)
    if camera is not None:
        return torch.from_numpy(out_3d.astype('float32')).cuda(), torch.from_numpy(out_camera.astype('float32')).cuda()
    else:
        return torch.from_numpy(out_3d.astype('float32')).cuda(), None

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n > 1e-3:
        print(n)
    return n < 1e-3

def rotationMatrixToEulerAngles(R) :

    R = np.asarray(R)
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793

    return np.array([x, y, z])

def calibrate_by_scale(predicted,target):
    assert predicted.shape == target.shape, print(predicted.shape,target.shape)
    predicted = np.asarray(predicted.cpu())
    target = np.asarray(target.cpu())
    # assert np.isnan(np.min(predicted)) == False
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale

    predicted_rescaled = a*predicted
    return torch.from_numpy(predicted_rescaled.astype('float32')).cuda()