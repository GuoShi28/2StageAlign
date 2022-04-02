import torch
### Paper models:
import models.archs.JDDB_BiGRU_Dconv_wInter as JDDB_BiGRU_Dconv_wInter
import pdb

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # pdb.set_trace()
    # image restoration
    if which_model == 'JDDB_BiGRU_Dconv_wInter':
        netG = JDDB_BiGRU_Dconv_wInter.JDDB_BiGRU(nf=opt_net['nf'],\
                groups=opt_net['groups'], in_channel=1, output_channel=3,\
                wInter=opt_net['w_Inter'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

def define_DS(opt):
    from models.archs.deep_down import CNN_downsampling
    VGG_model = CNN_downsampling(input_channels=4, kernel_size=3)
    return VGG_model


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
