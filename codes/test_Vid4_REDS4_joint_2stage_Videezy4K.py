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
import timeit

def read_rggb_img_seq_opts_joint(path, metadata):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    # img_l = [read_img(None, v) for v in img_path_l]
    img_l = []
    img_noise = []
    img_noise_map = []
    # pdb.set_trace()
    count = 1
    for v in img_path_l:
        print(v)
        temp = data_util.read_img(None, v)
        temp = data_util.BGR2RGB(temp)
        # pdb.set_trace()
        temp = torch.from_numpy(np.ascontiguousarray(temp))
        temp = temp.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
        
        temp, _ = unprocess.unprocess_meta_gt(temp, metadata['rgb_gain'], \
            metadata['red_gain'], \
            metadata['blue_gain'], metadata['rgb2cam'], \
            metadata['cam2rgb'])

        img_l.append(temp)
        shot_noise, read_noise = 6.4e-3, 2e-2
        # shot_noise, read_noise = 2.5e-3, 1e-2

        temp = unprocess.mosaic(temp)
        temp = unprocess.add_noise_test(temp, shot_noise, read_noise, count)
        count = count + 1
        temp = temp.clamp(0.0, 1.0)
        temp_np = torch.sqrt(shot_noise * temp + read_noise**2)

        img_noise.append(temp)
        img_noise_map.append(temp_np)

    # stack to Torch tensor
    imgs = torch.stack(img_l, axis=0)
    imgs_n = torch.stack(img_noise, axis=0)
    img_np = torch.stack(img_noise_map, axis=0)
    # pdb.set_trace()
    return imgs, imgs_n, img_np

def main():
    #################
    # configurations:
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = '4K_videos_new_HGain1_compare'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
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
    # import models.archs.JDDB_BiGRU_Dconv_wInter_woDConv as JDD_RNN_try
    # import models.archs.JDDB_BiGRU_Dconv_wInter_fRA as JDD_RNN_try
    model = JDD_RNN_try.JDDB_BiGRU(nf=64,\
                groups=8, in_channel=1, output_channel=3)
    model_path = '../experiments/J0007_JDDB_PBMNet_wgr_ncc_gt_s2h_re/models/550000_G.pth'
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    # ----- load VGG model -----
    # with 2 max pooling
    from models.archs.deep_down import CNN_downsampling
    DSNet_model = CNN_downsampling(input_channels=4, kernel_size=3)
    DSNet_path = '../experiments/J0007_JDDB_PBMNet_wgr_ncc_gt_s2h_re/models/550000_DSNet.pth'
    #### set up the models
    DSNet_model.load_state_dict(torch.load(DSNet_path), strict=True)
    DSNet_model.eval()
    DSNet_model = DSNet_model.to(device)

    #################---Step 2: set test dataset parameters---#######################
    ## 2.1 Metadata init
    xyz2cam = torch.FloatTensor([[1.0234, -0.2969, -0.2266],
                                [-0.5625, 1.6328, -0.0469],
                                [-0.0703, 0.2188, 0.6406]])
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(rgb2cam)
    # rgb_gains = 1.0 / 0.5
    rgb_gains = torch.FloatTensor([1.0])
    # rgb_gains = 1.0 
    # Red and blue gains represent white balance.
    red_gains  =  torch.FloatTensor([2.0])
    blue_gains =  torch.FloatTensor([1.7])
    metadata = {
        'cam2rgb': cam2rgb,
        'rgb2cam': rgb2cam,
        'rgb_gain': rgb_gains,
        'red_gain': red_gains,
        'blue_gain': blue_gains,
    }
    ## 2.2 Testing settings
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'
    # GT_dataset_folder = '../datasets/REDS4/GT'
    # GT_dataset_folder = '../datasets/Vid4/GT'
    GT_dataset_folder = '/home/guoshi/Data/4K_video_test/4K_frames_test'
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    print(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    ###  Compute flops if needed
    
    # util.print_model_parm_flops(model, (5, 2, 64, 64), input_num=1, cuda=True)
    _, n = util.get_network_description(model)
    logger.info('Network G with parameters: {:,d}'.format(n))
    
    #################---Step 3: begin testing---#######################
    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_ssim_center_l = []
    subfolder_name_l = []
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder_key, subfolder_GT in enumerate(subfolder_GT_l):
        subfolder_name = osp.basename(subfolder_GT_l[subfolder_key])
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)
        img_path_l = sorted(glob.glob(osp.join(subfolder_GT_l[subfolder_key], '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)
        #### 3.1 : read LQ and GT images
        img_GT_l, imgs_LQ, imgs_NMap = \
            read_rggb_img_seq_opts_joint(img_path_l, metadata)
        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        avg_ssim_center = 0
        for img_idx, img_path in enumerate(img_path_l):
            # img_idx = 20
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0)
            img_in_nmap = imgs_NMap.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0)
            start = timeit.default_timer()
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
            stop = timeit.default_timer()
            print('Time1: ', stop - start) 
            start = timeit.default_timer()
            #----> c. batch process
            # max_batch_num = (2*16)//((test_patch_size/64)**2)
            max_batch_num = 8
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
            stop = timeit.default_timer()
            print('Time2: ', stop - start) 
            start = timeit.default_timer()
            
            output_results = output_results[:, :, 0:int(2*H_ori), 0:int(2*W_ori)]
            #### 3.4 : transfer linear RGB to sRGB for visual pleasing
            output_results, ouput_show = process.process_test(output_results, metadata['red_gain'].unsqueeze(0), \
                metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0), \
                metadata['rgb_gain'].unsqueeze(0))
            output_results = output_results.squeeze().float().cpu().numpy()
            output_results = np.transpose(output_results, (1, 2, 0))
            otput_temp = output_results

            #### 3.5 : calculate PSNR
            GT = img_GT_l[img_idx:img_idx+1, :, :, :]
            GT, GT_show = process.process_test(GT, metadata['red_gain'].unsqueeze(0), \
                metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0), \
                metadata['rgb_gain'].unsqueeze(0))
            GT = GT.squeeze(0)
            GT = np.copy(GT)
            GT = np.transpose(GT, (1, 2, 0))
            
            if save_imgs:
                # output_results = GT
                output_results = np.clip(output_results, 0.0, 1.0)
                output_results = output_results[..., ::-1]
                output_results = (output_results * 255.0).round()
                output_results = output_results.astype(np.uint8)
                cv2.imwrite(osp.join(save_subfolder, '{}_Ours.png'.format(img_name)), output_results)
                
            otput_temp, GT = util.crop_border([otput_temp, GT], crop_border)
            crt_psnr = util.calculate_psnr(otput_temp * 255, GT * 255)
            crt_ssim = util.calculate_ssim(otput_temp * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB SSIM: {:.6f}'.format(img_idx + 1, img_name, crt_psnr, crt_ssim))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                avg_ssim_center += crt_ssim
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_ssim_center = avg_ssim_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_ssim_center_l.append(avg_ssim_center)
        avg_psnr_border_l.append(avg_psnr_border)
        #### 3.6 : Output test informations
        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Center SSIM: {:.6f} for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                    (N_center + N_border),
                                                                    avg_psnr_center, N_center,
                                                                    avg_ssim_center, N_center,
                                                                    avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border, ssim_center in zip(subfolder_name_l, avg_psnr_l,
                                                                avg_psnr_center_l, avg_psnr_border_l, avg_ssim_center_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Center SSIM: {:.6f}. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center, ssim_center,
                                                        psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB, center SSIM: {:.6f}. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_GT_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_ssim_center_l) / len(avg_ssim_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
