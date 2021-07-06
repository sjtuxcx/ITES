import torch
import numpy as np
from common.camera import calibrate_by_scale

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape, print(predicted.shape,target.shape)
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def calc_dist(predicted, target):
    assert len(predicted.shape) == 3 or len(predicted.shape) == 4
    if len(predicted.shape) == 3:
        dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=False), dim=len(target.shape) - 2, keepdim=False)
    else:
        dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=False), dim=2, keepdim=False)
    return np.asarray(dist.cpu())

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
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
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1),axis=1,keepdims=False)
    # return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))

def align_pose(predicted,target):
    assert predicted.shape == target.shape, print(predicted.shape,target.shape)
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
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
     # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    return predicted_aligned

def pck_auc(predicted, predicted_flip, target):
    eval_joint_in_h36_order = [2,3,5,6,7,8,9,10,11,12,13,14,15,16]
    predicted = np.asarray(predicted.cpu())
    predicted_flip = np.asarray(predicted_flip.cpu())
    target = np.asarray(target.cpu())
    # predicted_aligned = calibrate_by_scale(predicted,target)
    # predicted_aligned_flip = calibrate_by_scale(predicted_flip,target)
    predicted_aligned = predicted
    predicted_aligned_flip = predicted_flip
    predicted_aligned = predicted_aligned[:,eval_joint_in_h36_order,:]
    predicted_aligned_flip = predicted_aligned_flip[:,eval_joint_in_h36_order,:]
    target = target[:,eval_joint_in_h36_order,:]
    predicted_aligned = predicted_aligned.reshape(-1,3)
    predicted_aligned_flip = predicted_aligned_flip.reshape(-1,3) 
    target = target.reshape(-1,3)
    dists = np.linalg.norm(predicted_aligned-target,axis=1)
    dists_flip = np.linalg.norm(predicted_aligned_flip-target,axis=1)
    dist_best = np.minimum(dists,dists_flip)
    pck_150 = pck(dist_best)
    pck_values = np.arange(0.000,0.155,0.005)
    auc = 0
    for thres in pck_values:
        auc += pck(dist_best,threshold=thres) 
        # print(pck(dist_best,threshold=thres))
    auc = auc/len(pck_values)
    return pck_150, auc

def p_pck_auc(predicted, predicted_flip, target):
    eval_joint_in_h36_order = [2,3,5,6,7,8,9,10,11,12,13,14,15,16]
    predicted = np.asarray(predicted.cpu())
    predicted_flip = np.asarray(predicted_flip.cpu())
    target = np.asarray(target.cpu())
    predicted_aligned = align_pose(predicted,target)
    predicted_aligned_flip = align_pose(predicted_flip,target)
    predicted_aligned = predicted_aligned[:,eval_joint_in_h36_order,:]
    predicted_aligned_flip = predicted_aligned_flip[:,eval_joint_in_h36_order,:]
    target = target[:,eval_joint_in_h36_order,:]
    predicted_aligned = predicted_aligned.reshape(-1,3)
    predicted_aligned_flip = predicted_aligned_flip.reshape(-1,3) 
    target = target.reshape(-1,3)
    dists = np.linalg.norm(predicted_aligned-target,axis=1)
    dists_flip = np.linalg.norm(predicted_aligned_flip-target,axis=1)
    dist_best = np.minimum(dists,dists_flip)
    pck_150 = pck(dist_best)
    pck_values = np.arange(0.000,0.155,0.005)
    auc = 0
    for thres in pck_values:
        auc += pck(dist_best,threshold=thres) 
        # print(pck(dist_best,threshold=thres))
    auc = auc/len(pck_values)
    return pck_150, auc

def pck(dists, threshold=0.15):
    # predicted (N,3)
    return np.mean((dists < threshold)+0)

def p_mpjpe_mask(predicted, target, mask):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape, print(predicted.shape,target.shape)
    predicted = np.asarray(predicted.cpu())
    target = np.asarray(target.cpu())
    mask = np.asarray(mask.cpu())
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
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    # return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1),axis=1,keepdims=False)
    return (np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1)*mask).sum(1)/mask.sum(1)

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape, print(predicted.shape,target.shape)
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def consistency_loss(shape,target):
    # shape is a [N,K,J,3} tensor, K is number of shapes
    # target is a [N,J,3] tensor
    target = target[:, None, :,:]
    loss = torch.mean(torch.norm(shape - target, dim=len(target.shape)-1))
    return loss

def l1_loss(pre,gt):
    loss = torch.mean(torch.abs(pre-gt))
    return loss






