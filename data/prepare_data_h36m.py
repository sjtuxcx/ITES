
import argparse
import os
import zipfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree

import sys
sys.path.append('../')
from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates, camera_to_world
from common.utils import wrap
from common.h36m_dataset import h36m_cameras_extrinsic_params

output_filename = 'data_3d_h36m'
output_filename_2d = 'data_2d_h36m_gt'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')
    
    # Default: convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument('--from-archive', default='', type=str, metavar='PATH', help='convert preprocessed dataset')
    
    # Alternatively, convert dataset from original source (the Human3.6M dataset path must be specified manually)
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')
    
    args = parser.parse_args()
    
    if args.from_archive and args.from_source:
        print('Please specify only one argument')
        exit(0)
    
    # if os.path.exists(output_filename + '.npz'):
    #     print('The dataset already exists at', output_filename + '.npz')
    #     exit(0)
        
    if args.from_archive:
        print('Extracting Human3.6M dataset from', args.from_archive)
        with zipfile.ZipFile(args.from_archive, 'r') as archive:
            archive.extractall()
        
        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob('h36m/' + subject + '/MyPoses/3D_positions/*.h5')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]
                
                if subject == 'S11' and action == 'Directions':
                    continue # Discard corrupted video
                
                with h5py.File(f) as hf:
                    positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
                    positions /= 1000 # Meters instead of millimeters
                    output[subject][action] = positions.astype('float32')
        
        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)
        
        print('Cleaning up...')
        rmtree('h36m')
        
        print('Done.')
                
    elif args.from_source:
        print('Converting original Human3.6M dataset from', args.from_source)
        output = {}
        
        from scipy.io import loadmat

        id_order = ['54138969', '55011271', '58860488', '60457274']

        for subject in subjects:
            camera_info = h36m_cameras_extrinsic_params[subject]
            output[subject] = {}
            file_list = glob(args.from_source + '/' + subject + '/MyPoseFeatures/D3_Positions_mono/*.cdf.mat')
            assert len(file_list) == 120, "Expected 120 files for subject " + subject + ", got " + str(len(file_list))
            action_list = []
            for f in file_list:
                action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
                action = action.split('.')[0]
                if action not in action_list:
                    action_list.append(action)
            print(action_list)
            print(len(action_list))
            for action in action_list:
                position_list = []
                for idx,carema_id in enumerate(id_order):
                    f = args.from_source + '/' + subject + '/MyPoseFeatures/D3_Positions_mono/'+action+'.'+carema_id+'.cdf.mat'
                    # if subject == 'S11' and action == 'Directions':
                    #     continue # Discard corrupted video

                    # Use consistent naming convention

                    hf = loadmat(f)
                    positions = hf['data'][0, 0].reshape(-1, 32, 3)
                    positions /= 1000 # Meters instead of millimeters
                    positions_universal = camera_to_world(positions,R=np.array(camera_info[idx]['orientation']),t=np.array(camera_info[idx]['translation'])/1000)
                    position_list.append(positions_universal.astype('float32'))
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                                       .replace('WalkingDog', 'WalkDog')
                # if action == 'Directions 1':
                #     print('checking...')
                #     print(position_list[0]-position_list[1])
                #     print(position_list[1]-position_list[2])
                #     print(position_list[2]-position_list[3])
                output[subject][canonical_name] = sum(position_list)/4
                if action == 'Directions 1':
                    print(output[subject][canonical_name])
                    print(output[subject][canonical_name]-position_list[0])
        
        print('Saving...')
        #np.savez_compressed(output_filename, positions_3d=output)
        
        print('Done.')
        
    else:
        print('Please specify the dataset source')
        exit(0)
        
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    #np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    
    print('Done.')
