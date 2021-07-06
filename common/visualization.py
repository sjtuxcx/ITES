
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
from matplotlib import cm
import cv2
            
def draw_2d_pose(keypoints, skeleton, path):
    keypoints = np.asarray(keypoints.cpu())
    keypoints = keypoints - keypoints[0:1,:]
    keypoints[:,1:2] = -keypoints[:,1:2]
    # keypoints[:,0:1] = -keypoints[:,0:1]
    nkp = int(keypoints.shape[0])
    pid = np.linspace(0., 1., nkp)
    fig = plt.figure(1,figsize=(5,6))
    ax = fig.gca()
    # ax.set_xlim3d([-radius , radius])
    # ax.set_ylim3d([-radius, radius ])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'gray' if j in skeleton.joints_right() else 'orange'
        ax.plot([keypoints[j, 0], keypoints[j_parent, 0]],
                                    [keypoints[j, 1], keypoints[j_parent, 1]], linewidth=9,alpha=1,color=col)
    xs = keypoints[:,0]
    ys = keypoints[:,1]
    ax.scatter(xs, ys, s=80, c=pid, marker='o', cmap='gist_ncar',zorder=5)
    plt.axis('off')
    plt.savefig(path)
    plt.close()
    return

def draw_2d_img_and_pose(info,keypoints, skeleton, path):
    video = str(info[0])
    index = int(info[1])
    img = np.asarray(read_video(video,index))
    # frame = cv2.circle(frame,(kps[idx][i][0],kps[idx][i][1]),5,(0,0,255),-1)

    keypoints = np.asarray(keypoints.cpu())
    nkp = int(keypoints.shape[0])
    pid = np.linspace(0., 1., nkp)
    fig = plt.figure(1,figsize=(5,6))
    ax = fig.gca()
    # ax.set_xlim3d([-radius , radius])
    # ax.set_ylim3d([-radius, radius ])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'greenyellow' if j in skeleton.joints_right() else 'greenyellow'
        ax.plot([keypoints[j, 0], keypoints[j_parent, 0]],
                                    [keypoints[j, 1], keypoints[j_parent, 1]], linewidth=2.5,alpha=1,color=col)
    xs = keypoints[:,0]
    ys = keypoints[:,1]
    # ax.scatter(xs, ys, s=25, c='blue', marker='o', cmap='gist_ncar')
    ax.scatter(xs, ys, s=25, c='fuchsia', marker='o',zorder=2)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(path,dpi=40)
    plt.close()
    return

def draw_3d_pose(poses, skeleton, path):
    #poses n*3 dataset.skeleton()
    poses = poses - poses[0:1,:]
    poses = np.asarray(poses.cpu())
    nkp = int(poses.shape[0])
    pid = np.linspace(0., 1., nkp)
    poses[:,1:2] = -poses[:,1:2]
    poses[:,0:1] = -poses[:,0:1]
    plt.ioff()
    fig = plt.figure()
    radius = np.abs(poses).max()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=15, azim=70)
    ax.set_xlim3d([-radius , radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius ])
    # ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.patch.set_facecolor("white")  
    ax.dist = 7.5
    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'gray' if j in skeleton.joints_right() else 'orange'
        pos = poses
        ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 2], pos[j_parent, 2]],[pos[j, 1], pos[j_parent, 1]],linewidth=9,alpha=1,zdir='z', c=col)
    xs = poses[:,0]
    zs = poses[:,1]
    ys = poses[:,2]
    ax.scatter(xs, ys, zs, s=80, c=pid, marker='o', cmap='gist_ncar',zorder=2)
    # ax.scatter(xs, ys, zs, s=30, c='red', marker='o')
    plt.savefig(path,dpi=40)
    plt.close()
    return