'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import math
import h5py
import scipy.io as sio

import utils.util as util
import data.util as data_util
import utils.process as process
import utils.unprocess as unprocess
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import torchvision
import timeit
import matplotlib.pyplot as plt

def read_raw_img_seq(path):
    img_path_l = path
    img_l = []
    # pdb.set_trace()
    for idx, v in enumerate(img_path_l):
        print(v)
        img_denoise = h5py.File(v, 'r')
        buffer_np = np.float32(np.array(img_denoise['im']).T)
        buffer_np_4 = np.zeros((int(buffer_np.shape[0]/2), \
            int(buffer_np.shape[1]/2), 4), dtype=buffer_np.dtype)
        buffer_np_4[:,:,0] = buffer_np[0::2, 0::2]
        buffer_np_4[:,:,1] = buffer_np[0::2, 1::2]
        buffer_np_4[:,:,2] = buffer_np[1::2, 0::2]
        buffer_np_4[:,:,3] = buffer_np[1::2, 1::2]
        buffer_np_4 = np.clip(buffer_np_4, 0.0, 1.0)
        img_l.append(buffer_np_4)
        if idx == 2:
            meta_data = img_denoise['meta']
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float() # N * 4 * H * W

    return imgs, meta_data

def main():
    #################
    # configurations:
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    
    data_mode = 'real_test'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
   # Model settings
    N_in = 5
    stage = 2  # 1: for not using coarse alignment, 
               # 2: for the two stage alignment, if w_PBM = False, test in patch for one stage network.
    # for two stage alignment, user need to choose one of the coarse alignmnet method.
    LR_scale = 4 # perform coarse alignment on 1/LR_scale 
    test_patch_size = 128 # if using PBM for coase alignment, the test performs on patch domain
    patch_extend = 16
    save_imgs = True
    search_window = 128
    search_radius_lr = search_window//(LR_scale*2)
    ########################---Step 1: load model---###############################
    #### 1.2 model for generator
    import models.archs.JDDB_BiGRU_Dconv_wInter as JDD_RNN_try
    model = JDD_RNN_try.JDDB_BiGRU(nf=64,\
                groups=8, in_channel=1, output_channel=3)
    model_path = '../experiments/J0007_JDDB_PBMNet_wgr_ncc_gt_s2h/models/600000_G.pth'
    # ----- load VGG model -----
    # with 2 max pooling
    from models.archs.deep_down import CNN_downsampling
    DSNet_model = CNN_downsampling(input_channels=4, kernel_size=3)
    DSNet_path = '../experiments/J0007_JDDB_PBMNet_wgr_ncc_gt_s2h_re/models/550000_DSNet.pth'
    #### set up the models
    DSNet_model.load_state_dict(torch.load(DSNet_path), strict=True)
    DSNet_model.eval()
    DSNet_model = DSNet_model.to(device)
    
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    ## 2.2 Testing settings
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'
    test_dataset_folder = '/SC_burst/' # put dataset here
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    print(save_folder)
   
    #################---Step 3: begin testing---#######################
    subfolder_name_l = []
    subfolder_L_l = sorted(glob.glob(osp.join(test_dataset_folder, 'Scene*')))
    # for each subfolder
    for subfolder_key, subfolder_L in enumerate(subfolder_L_l):
        subfolder_name = osp.basename(subfolder_L)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder_L, '*.mat')))
        img_path_l = img_path_l[0:N_in]
        if save_imgs:
            util.mkdirs(save_subfolder)
       
        #### read LQ and GT images
        imgs_LQ, metadata = read_raw_img_seq(img_path_l)

        ### read meta data
        ColorMatrix = np.array(metadata['ColorMatrix2'])
        WhiteParam = np.array(metadata['AsShotNeutral'])
        UnknoTag = metadata['UnknownTags']
        TagValue = UnknoTag['Value']
        TagNoise = TagValue[0,7]
        NoisePara = np.array(UnknoTag[TagNoise])
        shot_noise = torch.from_numpy(NoisePara[0]).float()
        read_noise = torch.from_numpy(np.sqrt(NoisePara[1])).float()
        ## Metadata
        xyz2cam = torch.FloatTensor(ColorMatrix)
        xyz2cam = torch.reshape(xyz2cam, (3, 3))
        rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
        
        rgb2cam = torch.mm(xyz2cam, rgb2xyz)
        # Normalizes each row.
        rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
        cam2rgb = torch.inverse(rgb2cam)
        rgb_gains = torch.FloatTensor([1.0])
        # Red and blue gains represent white balance.
        red_gains  =  torch.FloatTensor(1.0/WhiteParam[0])
        blue_gains =  torch.FloatTensor(1.0/WhiteParam[2])
        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gains,
            'red_gain': red_gains,
            'blue_gain': blue_gains,
        }

        W = imgs_LQ.shape[2]
        H = imgs_LQ.shape[3]
        
        imgs_in = imgs_LQ # T, C(4), H, W
        imgs_in = imgs_in.view(1, imgs_in.shape[0], imgs_in.shape[1], imgs_in.shape[2], imgs_in.shape[3])
        img_in_nmap = torch.sqrt(shot_noise * imgs_in + read_noise**2)
       
        #### 3.2 : pad image to ensure the size of test image can divide by need_scale
        need_scale = test_patch_size
        imgs_in, H_ori, W_ori, H_new, W_new = util.pad_img_2_setscale(imgs_in, need_scale)
        img_in_nmap,_,_,_,_ = util.pad_img_2_setscale(img_in_nmap, need_scale)
        imgs_in = imgs_in.cuda()
        img_in_nmap = img_in_nmap.cuda()
        B_num, N_num, C_num,_,_ = img_in_nmap.size()
        #### 3.3 : perform two-stage alignmnet and resconstraction
        #### Note: performe coarse alignment on LR image domain
        #----> a. pad image to avoid the edge problem
        # patch_extend = self.train_patch_size//16
        imgs_in_pad = imgs_in.cpu().clone().numpy()
        imgs_in_nmap_pad = img_in_nmap.cpu().clone().numpy()
        imgs_in_pad = np.pad(imgs_in_pad, ((0, 0), (0, 0), (0, 0), \
            (patch_extend, patch_extend), (patch_extend, patch_extend)), \
                'constant')
        imgs_in_nmap_pad = np.pad(imgs_in_nmap_pad, ((0, 0), (0, 0), (0, 0), \
            (patch_extend, patch_extend), (patch_extend, patch_extend)), \
                'constant')
        imgs_in_pad = torch.from_numpy(imgs_in_pad).cuda()
        imgs_in_nmap_pad = torch.from_numpy(imgs_in_nmap_pad).cuda()
        #----> b. use PBM and output all aligned patches
        DSNet_net = DSNet_model
        all_img_in_patches, all_nmap_in_patches, patch_num, h_num, w_num = \
                util.caligned_wPBM_test(imgs_in_pad, \
                imgs_in_nmap_pad, LR_scale, test_patch_size, patch_extend, \
                search_radius_lr, DSNet_net)
        #----> c. batch process
        max_batch_num = 4
        output_patches = util.batch_forward(model, \
            all_img_in_patches, all_nmap_in_patches, patch_num, max_batch_num)
        #----> d. merge back to full-size image
        # init output
        output_results = torch.zeros((1, 3, int(imgs_in.shape[3]*2), \
            int(imgs_in.shape[4]*2)))
        output_results = output_results.float()
        # pdb.set_trace()
        output_results = util.merge_back(output_patches, output_results, \
            h_num, w_num, test_patch_size, patch_extend)
        
        output_results = output_results[:, :, 0:int(2*H_ori), 0:int(2*W_ori)]
        #### 3.4 : transfer linear RGB to sRGB for visual pleasing
        output_results, ouput_show = process.process_test(output_results, metadata['red_gain'].unsqueeze(0), \
            metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0), \
            metadata['rgb_gain'].unsqueeze(0))
        output_results = output_results.squeeze().float().cpu().numpy()
        output_results = np.transpose(output_results, (1, 2, 0))
        otput_temp = output_results

        if save_imgs:
            output_results = np.clip(output_results, 0.0, 1.0)
            output_results = output_results[..., ::-1]
            output_results = (output_results * 255.0).round()
            output_results = output_results.astype(np.uint8)
            cv2.imwrite(osp.join(save_subfolder, 'Ours.png'), output_results)

if __name__ == '__main__':
    main()
