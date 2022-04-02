'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from utils import unprocess
import matplotlib.pyplot as plt
import pdb
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class REDSDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        # self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        # self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove the REDS4 for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        ]
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        #self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
        #                        meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)
   

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        ### Random cut into square patches
        m_size = 720
        rnd_w = random.randint(0, max(0, 1280 - m_size))
        #### get the GT image (as the center frame)
        img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        img_GT = img_GT[:,rnd_w:rnd_w+m_size, :]
        #-----------------
        img_GT = util.BGR2RGB(img_GT) # RGB
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT))
        img_GT = img_GT.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
        img_GT, metadata = unprocess.unprocess_gt(img_GT)
        # img_GT = img_GT.permute(1, 2, 0) # Re-Permute the tensor back HxWxC format
        
        ## Random noise level
        shot_noise, read_noise = unprocess.random_noise_levels_kpn()

        #### get LQ images
        LQ_size_tuple = (3, 720, 1280)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ = util.read_img(self.GT_env, '{}_{:08d}'.format(name_a, v), LQ_size_tuple)
            img_LQ = img_LQ[:,rnd_w:rnd_w+m_size, :]
            img_LQ = util.BGR2RGB(img_LQ) # RGB
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ))
            img_LQ = img_LQ.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
            img_LQ, _ = unprocess.unprocess_meta_gt(img_LQ, metadata['rgb_gain'], \
                metadata['red_gain'], \
                metadata['blue_gain'], metadata['rgb2cam'], metadata['cam2rgb'])
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            img_LQ_l.append(img_GT)
            # augmentation - flip, rotate
            rlt = util.augment2(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

            img_LQ_l = [unprocess.mosaic(v) for v in img_LQ_l]
            img_GT_raw = img_LQ_l
            img_LQ_l = [unprocess.add_noise(v, shot_noise, read_noise) for v in img_LQ_l]
            img_LQ_l = [v.clamp(0.0, 1.0) for v in img_LQ_l]
            img_noise_map = [shot_noise * v + read_noise for v in img_LQ_l]

        # stack LQ images to NHWC, N is the frame number
        img_GT_raws = torch.stack(img_GT_raw, axis=0)
        img_LQs = torch.stack(img_LQ_l, axis=0)
        img_NMaps = torch.stack(img_noise_map, axis=0)
        # pdb.set_trace()
        inputs = {'LQs': img_LQs, 'NMaps': img_NMaps, 'GT': img_GT, 'key': key, 'GT_raw': img_GT_raws}
        inputs.update(metadata)

        return inputs

    def __len__(self):
        return len(self.paths_GT)
