import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, TVLoss, photometric_reconstruction_loss, explainability_loss, smooth_loss, correlation_loss, sRGBforward, color_loss
import numpy as np
import pdb

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

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.tv_pix = TVLoss().to(self.device)
            #### optimizers
            # pdb.set_trace()
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if opt['network_G']['which_model_G'] == 'UNet_SpyNet':
                self.tv_pix = TVLoss().to(self.device)
                spy_s1_params = []
                spy_s2_params = []
                spy_s3_params = []
                spy_s4_params = []
                spy_s5_params = []
                normal_params = []
                for k, v in self.netG.named_parameters():
                    # pdb.set_trace()
                    if v.requires_grad:
                        if 'spynet_s1' in k:
                            spy_s1_params.append(v)
                        elif 'spynet_s2' in k:
                            spy_s2_params.append(v)
                        elif 'spynet_s3' in k:
                            spy_s3_params.append(v)
                        elif 'spynet_s4' in k:
                            spy_s4_params.append(v)
                        elif 'spynet_s5' in k:
                            spy_s5_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s1_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s2_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s3_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s4_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s5_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            elif opt['network_G']['which_model_G'] == 'UNet_SpyNetPro':
                self.tv_pix = TVLoss().to(self.device)
                spy_s1_params = []
                spy_s2_params = []
                spy_s3_params = []
                spy_s4_params = []
                spy_s5_params = []
                normal_params = []
                for k, v in self.netG.named_parameters():
                    # pdb.set_trace()
                    if v.requires_grad:
                        if 'spynet_s1' in k:
                            spy_s1_params.append(v)
                        elif 'spynet_s2' in k:
                            spy_s2_params.append(v)
                        elif 'spynet_s3' in k:
                            spy_s3_params.append(v)
                        elif 'spynet_s4' in k:
                            spy_s4_params.append(v)
                        elif 'spynet_s5' in k:
                            spy_s5_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s1_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s2_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s3_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s4_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': spy_s5_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            elif opt['network_G']['which_model_G'] == 'UNet_Sfm':
                sfm_params = []
                normal_params = []
                # pdb.set_trace()
                for k, v in self.netG.named_parameters():
                    # print(k)
                    # pdb.set_trace()
                    if v.requires_grad:
                        if ('disp_net' in k) or ('pose_exp_net' in k):
                            sfm_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': sfm_params,
                        'lr': train_opt['lr_G']
                    },
                ]

            else:
                if train_opt['ft_tsa_only']:
                    normal_params = []
                    tsa_fusion_params = []
                    for k, v in self.netG.named_parameters():
                        if v.requires_grad:
                            if 'tsa_fusion' in k:
                                tsa_fusion_params.append(v)
                            else:
                                normal_params.append(v)
                        else:
                            if self.rank <= 0:
                                logger.warning('Params [{:s}] will not optimize.'.format(k))
                    optim_params = [
                        {  # add normal params first
                            'params': normal_params,
                            'lr': train_opt['lr_G']
                        },
                        {
                            'params': tsa_fusion_params,
                            'lr': train_opt['lr_G']
                        },
                    ]
                else:
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
            # pdb.set_trace()

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
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        # pdb.set_trace()
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_rgb(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        # pdb.set_trace()
        l_pix = self.l_pix_w * self.cri_pix(sRGBforward(self.fake_H), sRGBforward(self.real_H))
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_rgb_color(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        # pdb.set_trace()
        fake_H_sRGB = sRGBforward(self.fake_H)
        real_H_sRGB = sRGBforward(self.real_H)
        # pdb.set_trace()
        l_pix = self.l_pix_w * self.cri_pix(fake_H_sRGB, real_H_sRGB)
        # l_pix += color_loss(fake_H_sRGB, real_H_sRGB) 
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_rgb_color_rnn(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        # pdb.set_trace()
        fake_H_sRGB = sRGBforward(self.fake_H)
        real_H_sRGB = sRGBforward(self.real_H)
        n_frame = fake_H_sRGB.shape[1]
        center_frame = int((n_frame-1)/2)
        l_pix = 0
        for n in range(n_frame):
            temp = self.l_pix_w * self.cri_pix(fake_H_sRGB[:,n,:,:,:], real_H_sRGB)
            if n == center_frame:
                l_pix = l_pix + temp
            else:
                l_pix = l_pix + 0.0 * temp
        
        # l_pix += color_loss(fake_H_sRGB, real_H_sRGB) 
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_rgb_color_wrap(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H, self.optical = self.netG(self.var_L)

        # pdb.set_trace()
        fake_H_sRGB = sRGBforward(self.fake_H)
        real_H_sRGB = sRGBforward(self.real_H)
        l_pix = self.l_pix_w * self.cri_pix(fake_H_sRGB, real_H_sRGB)
        # pdb.set_trace()
        for s in range(5):
            l_pix = l_pix + 0.1*self.tv_pix(self.optical[s])
        # l_pix += color_loss(fake_H_sRGB, real_H_sRGB) 
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_color(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        # pdb.set_trace()
        # fake_H_sRGB = sRGBforward(self.fake_H)
        # real_H_sRGB = sRGBforward(self.real_H)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix += color_loss(self.fake_H, self.real_H) 
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters_EDVRPro(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H, self.corr_features, self.gt_features = self.netG(self.var_L)
        # pdb.set_trace()
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        #print(l_pix)
        ann_para = 0.01 * (0.9998**step)
        l_corr = ann_para*correlation_loss(self.corr_features, self.gt_features)
        #print(l_corr)
        l_pix += l_corr
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters_KPN(self, step):
        self.optimizer_G.zero_grad()
        self.fake_ALL, self.fake_H = self.netG(self.var_L)
        # avg loss
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        # ann_loss
        ann_para = 100 * (0.9998**step)
        frame_num = len(self.fake_ALL)
        for fr in range(frame_num):
            l_pix = l_pix + ann_para*self.cri_pix(self.fake_ALL[fr], self.real_H)

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_Sfm(self, step):
        # print('here')
        self.optimizer_G.zero_grad()
        self.fake_H, wrapped_image_scale, tgt_image_scale, valid_points_scale, explainability_mask, pose, depth = self.netG(self.var_L)
        
        if step < 50000: 
            loss_para = np.array([1,0])
        else:
            loss_para = np.array([1,1])
            self.optimizers[0].param_groups[1]['lr'] = 0
          
        # avg loss
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        # photometric_reconstruction_loss
        l_ph = photometric_reconstruction_loss(wrapped_image_scale, tgt_image_scale, valid_points_scale, explainability_mask)
        l_smooth = smooth_loss(depth)
        l_ex = explainability_loss(explainability_mask)

        l_pix = loss_para[1]*l_pix + loss_para[0]*(l_ph + 0.2*l_ex + 0.1*l_smooth)

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_SpyNet(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H, self.aligned, self.gt, self.flow = self.netG(self.var_L)
        # output: multi_list, len(multi_list)=5, each, B*(F-1),C,H,W
        if step < 50000:
            ### Train SpyNet
            if step < 10000:
                loss_para = np.array([1,0,0,0,0])
                    
            elif step < 20000:
                loss_para = np.array([0,1,0,0,0])
                self.optimizers[0].param_groups[1]['lr'] = 0
                # pdb.set_trace()
            elif step < 30000:
                loss_para = np.array([0,0,1,0,0])
                self.optimizers[0].param_groups[1]['lr'] = 0
                self.optimizers[0].param_groups[2]['lr'] = 0
            elif step < 40000:
                loss_para = np.array([0,0,0,1,0])
                self.optimizers[0].param_groups[1]['lr'] = 0
                self.optimizers[0].param_groups[2]['lr'] = 0
                self.optimizers[0].param_groups[3]['lr'] = 0
            else:
                loss_para = np.array([0,0,0,0,1])
                self.optimizers[0].param_groups[1]['lr'] = 0
                self.optimizers[0].param_groups[2]['lr'] = 0
                self.optimizers[0].param_groups[3]['lr'] = 0
                self.optimizers[0].param_groups[4]['lr'] = 0
            avg_loss = 0
        else:
            ### Train UNet
            self.optimizers[0].param_groups[1]['lr'] = 0
            self.optimizers[0].param_groups[2]['lr'] = 0
            self.optimizers[0].param_groups[3]['lr'] = 0
            self.optimizers[0].param_groups[4]['lr'] = 0
            self.optimizers[0].param_groups[5]['lr'] = 0
            loss_para = np.array([0,0,0,0,0])
            avg_loss = 1

        # avg loss
        l_pix = avg_loss * self.cri_pix(self.fake_H, self.real_H)
        # Spy loss
        scale_num = len(self.aligned)
        for s in range(scale_num):
            l_pix = l_pix + loss_para[s]*self.cri_pix(self.aligned[s], self.gt[s])

        for s in range(scale_num):
            l_pix = l_pix + loss_para[s]*self.tv_pix(self.flow[s])

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
    
    def optimize_parameters_SpyNetPro(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H, self.aligned, self.gt, self.flow = self.netG(self.var_L)
        # output: multi_list, len(multi_list)=5, each, B*(F-1),C,H,W
        # step = 80000
        if step < 100000:
            ### Train SpyNet
            if step < 20000:
                loss_para = np.array([1,0,0,0,0])     
            elif step < 40000:
                loss_para = np.array([1,1,0,0,0])
            elif step < 60000:
                loss_para = np.array([1,1,1,0,0])
            elif step < 80000:
                loss_para = np.array([1,1,1,1,0])
            else:
                loss_para = np.array([1,1,1,1,1])
            avg_loss = 0
        else:
            ### Train UNet
            self.optimizers[0].param_groups[1]['lr'] = 0
            self.optimizers[0].param_groups[2]['lr'] = 0
            self.optimizers[0].param_groups[3]['lr'] = 0
            self.optimizers[0].param_groups[4]['lr'] = 0
            self.optimizers[0].param_groups[5]['lr'] = 0
            loss_para = np.array([0,0,0,0,0])
            avg_loss = 1

        # avg loss
        l_pix = avg_loss * self.cri_pix(self.fake_H, self.real_H)
        # Spy loss
        scale_num = len(self.aligned)
        align_loss = 0
        for s in range(scale_num):
            align_loss = align_loss + loss_para[s]*self.cri_pix(self.aligned[s], self.gt[s])

        tv_loss = 0
        for s in range(scale_num):
            tv_loss = tv_loss + 5*loss_para[s]*self.tv_pix(self.flow[s])
        
        #print(align_loss)
        #print(tv_loss)
        l_pix = l_pix + align_loss + tv_loss
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_KPN(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_ALL, self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_SpyNet(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.aligned, self.gt = self.netG(self.var_L)
        self.netG.train()

    def test_SfmNet(self):
        self.netG.eval()
        with torch.no_grad():
            #self.fake_H, self.aligned, self.gt = self.netG(self.var_L)
            self.fake_H, self.aligned, self.validate, self.explain, self.pose, self.depth = self.netG(self.var_L)
        # pdb.set_trace()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_sfm(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['aligned'] = self.aligned[0].detach()[0].float().cpu()
        out_dict['refer'] = self.real_H.detach()[0].float().cpu()
        out_dict['exp'] = self.explain.detach()[0,0:1,:,:].float().cpu()
        out_dict['depth'] = self.depth.detach().float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict
    
    def get_current_visuals_spy(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['aligned'] = self.aligned[0][1].detach()[0].float().cpu()
        out_dict['refer'] = self.gt[0].detach()[0].float().cpu()
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

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
