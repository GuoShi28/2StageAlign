import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks_2stage as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierMaskLoss
import numpy as np
import pdb
from utils import process, unprocess
import utils.util as util
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.archs.arch_util import flow_warp
import random
import data.util as data_util

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.train_opt = train_opt
        self.train_patch_size = opt['datasets']['train']['LQ_size']
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netDS = networks.define_DS(opt).to(self.device) # define vgg network

        opt_net = opt['network_G']
        self.align_scale = opt_net['align_scale']
        self.w_interL = opt_net['w_Inter']
        self.search_window = opt_net['search_window']
        self.search_radius_lr = self.search_window//(self.align_scale*2)
        
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netDS = DistributedDataParallel(self.netDS, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.netDS = DataParallel(self.netDS)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            self.netDS.train()
            #### loss
            self.cri_pix_mask = CharbonnierMaskLoss().to(self.device)
            self.cri_pix = CharbonnierLoss().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            
            #----- optimizer for generator G -------
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #----- optimizer for lr deep feature extractor -------
            optim_params = []
            for k, v in self.netDS.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_DSNet = torch.optim.Adam(optim_params, lr=train_opt['lr_DSNet'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_DSNet)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.var_NM = data['NMaps'].to(self.device)
        self.red_gains = data['red_gain'].to(self.device)
        self.blue_gains = data['blue_gain'].to(self.device)
        self.cam2rgbs = data['cam2rgb'].to(self.device)

        if need_GT:
            self.real_H = data['GT'].to(self.device)
            self.real_raw = data['GT_raw'].to(self.device)

    def set_params_lr_zero(self):
        self.optimizer_G.param_groups[0]['lr'] = 0
        self.optimizer_DSNet.param_groups[0]['lr'] = 0
    
    def optimize_parameters_rgb(self, step):
        self.optimizer_G.zero_grad()
        self.optimizer_DSNet.zero_grad()
        # Step 1: prealign images in small scale using PBM
        #         And random select image patch used to train network G 
        aligned_x = []
        aligned_nmap = []
        small_scale = self.align_scale
        if small_scale<0:   small_scale=1
        B, N, C, H, W = self.var_L.size()  # N video frames, C is 4 response to RGGB channel
        center_idx = N//2
        # randomly select patch position
        LQ_size = self.train_patch_size
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - 2*LQ_size -1))

        temp_patch = self.var_L[:,:,:,rnd_h:rnd_h+LQ_size,rnd_w:rnd_w+LQ_size].clone()
        temp_nm_patch = self.var_NM[:,:,:,rnd_h:rnd_h+LQ_size,rnd_w:rnd_w+LQ_size].clone()
        # ----- obtain GT one-hot vector -----
        patch_raw_gt = self.real_raw[:,:,:,rnd_h:rnd_h+LQ_size,rnd_w:rnd_w+LQ_size].clone()
        
        ## lr_features on full image
        lr_features = util.cal_lr_fea(self.var_L, self.netDS)
        image_patch_new, nmap_patch_new, ncc_scores, ncc_scores_nor, one_hot_gt = \
            util.search_patch_NCC_2d_pymaid_wDSNet_wE2E(temp_patch, temp_nm_patch, \
                self.var_L.clone(), self.var_NM.clone(), lr_features,\
                rnd_h, rnd_w, small_scale, self.search_radius_lr, \
                patch_raw_gt, self.real_raw, step)

        aligned_x = image_patch_new
        aligned_nmap = nmap_patch_new

        # --- Step 2: update the image reconstraction network G
        self.fake_H = self.netG(aligned_x, aligned_nmap)

        # --- Step 3: Calculate loss
        # Step 3.1: collect GT patch:
        GT_patch = self.real_H[:,:,2*rnd_h:2*(rnd_h+LQ_size),2*rnd_w:2*(rnd_w+LQ_size)]
        ## high frequency mask
        if self.w_interL:
            real_mask = data_util.edge_map_generate(GT_patch)

        # Step 3.2: obtain sRGB image using ISP operator
        fake_H_rgb = process.process_train(self.fake_H, self.red_gains, \
            self.blue_gains, self.cam2rgbs)
        real_H_rgb = process.process_train(GT_patch, self.red_gains, \
            self.blue_gains, self.cam2rgbs)
        # Step 3.3: calculate pixel-wise reconstraction loss in both linear and sRGB space
        # if needed: also winterpolation loss
        if self.w_interL == False:
            l_pix = self.cri_pix(self.fake_H, GT_patch) + \
                self.cri_pix(fake_H_rgb, real_H_rgb)
        else:
            B = self.fake_H.shape[0]//2
            # pixel_wise loss
            l_pix = self.cri_pix(self.fake_H[0:B,...], GT_patch) + \
                self.cri_pix(fake_H_rgb[0:B,...], real_H_rgb)
            # interpolation loss
            l_pix = l_pix + self.cri_pix_mask(self.fake_H[B:,...], GT_patch, real_mask) + \
                self.cri_pix_mask(fake_H_rgb[B:,...], real_H_rgb, real_mask)
        
        # --- Add Constrain ---
        weight_de = 1e5
        for i in range(len(ncc_scores)):
            length = np.float(ncc_scores_nor[i].shape[0])
            sum_ncc = torch.sum(ncc_scores_nor[i])
            l_pix = l_pix + weight_de*torch.abs(sum_ncc - 1.0)
            var_ncc = torch.var(ncc_scores_nor[i])
            l_pix = l_pix + weight_de*torch.abs(var_ncc - 1.0/length)

            # ---- cal with GT
            if step < 200000:
                l_pix = l_pix + 1e3*self.cri_pix(one_hot_gt[i], ncc_scores[i])
        l_pix.backward()
        
        # pdb.set_trace()
        self.optimizer_G.step()
        self.optimizer_DSNet.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def test(self):
        self.netG.eval()
        self.netDS.eval()
        with torch.no_grad():
            #### 3.2 : pad image to ensure the size of test image can divide by need_scale
            need_scale = self.train_patch_size
            imgs_in = self.var_L.clone()
            img_in_nmap = self.var_NM.clone()
            imgs_in, H_ori, W_ori, H_new, W_new = util.pad_img_2_setscale(imgs_in, need_scale)
            img_in_nmap,_,_,_,_ = util.pad_img_2_setscale(img_in_nmap, need_scale)
            imgs_in = imgs_in.cuda()
            img_in_nmap = img_in_nmap.cuda()
            B_num, N_num, C_num,_,_ = img_in_nmap.size()
            #### 3.3 : perform two-stage alignmnet and resconstraction
            #### Note: performe coarse alignment on LR image domain
            LR_scale = self.align_scale
            #----> a. pad image to avoid the edge problem
            # patch_extend = self.train_patch_size//16
            patch_extend = 0
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
            all_img_in_patches, all_nmap_in_patches, patch_num, h_num, w_num = \
                util.caligned_wPBM_wDSNet(imgs_in_pad, \
                imgs_in_nmap_pad, LR_scale, self.train_patch_size, patch_extend, \
                self.search_radius_lr, self.netDS)
            #----> c. batch process
            max_batch_num = (4*16)//((self.train_patch_size/64)**2)
            output_patches = util.batch_forward(self.netG, \
                all_img_in_patches, all_nmap_in_patches, patch_num, max_batch_num)
            #----> d. merge back to full-size image
            # init output
            output_results = torch.zeros((1, 3, int(imgs_in.shape[3]*2), \
                int(imgs_in.shape[4]*2)))
            output_results = output_results.float()
            # pdb.set_trace()
            output_results = util.merge_back(output_patches, output_results, \
                h_num, w_num, self.train_patch_size, patch_extend)
            
            output_results = output_results[:, :, 0:int(2*H_ori), 0:int(2*W_ori)]
            self.fake_H = output_results
        self.netG.train()
        self.netDS.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        
        s, n = self.get_network_description(self.netDS)
        if isinstance(self.netDS, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netDS.__class__.__name__,
                                            self.netDS.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netDS.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network DSNet structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        
        load_path_DS = self.opt['path']['pretrain_model_DSNet']
        if load_path_DS is not None:
            logger.info('Loading model for DSNet [{:s}] ...'.format(load_path_VGG))
            self.load_network(load_path_DS, self.netDS, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netDS, 'DSNet', iter_label)
