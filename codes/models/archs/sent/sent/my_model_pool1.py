#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:02:16 2019

@author: justin
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../')
import layers

def lrelu(x):
    return tf.maximum(x*0.2,x)
def relu(x):
    return tf.nn.relu(x)

def upsample_to(x1, x2, output_channels, in_channels, scope,reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        pool_size = 2
        deconv_filter = tf.get_variable(shape= [pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.001),name='dcf')
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    return deconv

def upsample_and_concat_c(x1, x2, output_channels, in_channels, scope,reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        pool_size = 2
        deconv_filter = tf.get_variable(shape= [pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.001),name='dcf')
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

        deconv_output =  tf.concat([deconv, x2],3)
   #     deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def est_structure(x,size,sigma):
    ## x is a single channel tensor
    def _tf_fspecial_gauss(size, sigma):
        x_data, y_data = np.mgrid[-size//2 - 1:size//2 + 1, -size//2 - 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2.0 + y**2.0)/(2.0*sigma**2.0)))
        return g / tf.reduce_sum(g)

    window = _tf_fspecial_gauss(size, sigma)
    final = tf.nn.conv2d(x, window, strides=[1,1,1,1], padding='SAME')
    return final

def pad(x,p=1):
    p = int(p)
    return tf.pad(x,[[0,0],[p,p],[p,p],[0,0]],'REFLECT')
def pad_4(x,p=1):
    p=int(p)
    return tf.pad(x,[[0,0],[p,p],[p,p],[0,0],[0,0]],'REFLECT')

def Conv_block(input,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,chan=32,act=lrelu,name=None,reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        current = input
        for i in range(num_conv):
            current=slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),chan,[fil_s,fil_s], 
                            rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv%d'%(i),reuse=reuse)
        out = slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),num_out,[fil_s,fil_s], 
                            rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv_out',reuse=reuse)  
    return out

def Conv_block1(input,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,chan=32,act=lrelu,name=None,reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        current = input
        for i in range(num_conv-1):
            current=slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),chan,[fil_s,fil_s], 
                            rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv%d'%(i),reuse=reuse)
        out = slim.conv2d(pad(current,(fil_s-1)/2),num_out,[fil_s,fil_s], 
                             activation_fn=act,padding='VALID',scope='g_conv_out',reuse=reuse)  
    return out

def Conv_block_residual(input,num_block=1,rate=[1]*10,fil_s=3,chan=32,act=lrelu,name=None,reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        current = slim.conv2d(pad(input,(fil_s-1)/2),chan,[fil_s,fil_s], activation_fn=act,padding='VALID',scope='g_conv',reuse=reuse)
        for i in range(num_block):
            add = current
            current=slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),chan,[fil_s,fil_s], 
                            rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv01%d'%(i),reuse=reuse)
            current=slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),chan,[fil_s,fil_s], 
                            rate=rate[i], activation_fn=None,padding='VALID',scope='g_conv02%d'%(i),reuse=reuse)
            current = act(add + current)
        out = current
    return out
    
    

def U_net22(input,num_down=4,num_block=1,num_conv=1,num_out=3,multis=False,rate=[1]*10,fil_s=3,w_init=None,b_init=None,is_residual=False,start_chan=32,act=lrelu,is_global=False,name=None,reuse=False):
    ## parameters
    conv_ = []
    chan_ = []
    if w_init is None:
        w_init = tf.contrib.slim.xavier_initializer()
    if b_init is None:
        b_init = tf.constant_initializer(value=0.0)    
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))
    
    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(pad(current,(fil_s-1)/2),chan_[i],[fil_s,fil_s], 
                                      weights_initializer=w_init,biases_initializer=b_init,
                                      activation_fn=act,scope='g_conv%d'%(i),padding='VALID',reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(pad(current,rate[i]*(fil_s-1)/2),chan_[i],[fil_s,fil_s], 
                                             weights_initializer=w_init,biases_initializer=b_init,rate=rate[i], activation_fn=act,padding='VALID',scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current,(fil_s-1)/2),chan_[i],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init, activation_fn=None,padding='VALID',scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)     
                    else:
                        current = act(current)    
                #pool=slim.conv2d(current,chan_[i],[fil_s,fil_s], stride=2,
                                       # weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='SAME',scope='pool%d'%(i),reuse=reuse)        
                pool=slim.max_pool2d(current, [2, 2], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(pad(current,(fil_s-1)/2),chan_[num_down],[fil_s,fil_s], 
                                weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='VALID',scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(pad(current,rate[num_down]*(fil_s-1)/2),chan_[num_down],[fil_s,fil_s],
                                          weights_initializer=w_init,biases_initializer=b_init, rate=rate[num_down],activation_fn=act,padding='VALID',scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(pad(current,(fil_s-1)/2),chan_[num_down],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init,activation_fn=None,padding='VALID',scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
                else:
                    current = act(current) 
            restore_temp = current

        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
                    current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
                '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=lrelu,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=None,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])
        
        multis_list = []
        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(pad(current,(fil_s-1)/2),chan_[index_current],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, padding='VALID',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(pad(current,rate[index_current]*(fil_s-1)/2), chan_[index_current],[fil_s,fil_s], 
                                            weights_initializer=w_init,biases_initializer=b_init,rate=rate[index_current], padding='VALID',activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(pad(current,(fil_s-1)/2),  chan_[index_current],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init, padding='VALID',activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if i == num_down-1 and ii == num_block-1:
                        if is_residual is True:
                            current = current + adding
                        else:
                            current = current
                    else:
                        if is_residual is True:
                            current = act(current + adding)
                        else:
                            current = act(current) 
            '''
            if multis is True:
                multis_list.append(slim.conv2d(current, num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=tf.nn.tanh,scope='final',reuse=reuse))    
            '''    
            final = slim.conv2d(current,  num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=None,scope='final',reuse=reuse)

    return final



def U_net222(input,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,is_residual=False,start_chan=32,act=lrelu,is_global=False,name=None,reuse=False):
    ## parameters
    conv_ = []
    chan_ = []
    if w_init is None:
        w_init = tf.contrib.slim.xavier_initializer()
    if b_init is None:
        b_init = tf.constant_initializer(value=0.0)    
    for i in range(num_down+1):
        chan_.append(start_chan*(2**(i)))
    
    with tf.variable_scope(name,reuse=reuse):
        current = input
        with tf.variable_scope('contracting_ops',reuse=reuse):
            for i in range(num_down):
                current = slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                      weights_initializer=w_init,biases_initializer=b_init,
                                      activation_fn=act,scope='g_conv%d'%(i),padding='SAME',reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                         current=slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                             weights_initializer=w_init,biases_initializer=b_init,rate=rate[i], activation_fn=act,padding='SAME',scope='g_conv%d_block%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init, activation_fn=None,padding='SAME',scope='g_conv%d_block%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)     
                    else:
                        current = act(current)    
                pool=slim.conv2d(current,chan_[i],[fil_s,fil_s], stride=2,
                                        weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='SAME',scope='pool%d'%(i),reuse=reuse)        
                #pool=slim.max_pool2d(current, [2, 2], padding='SAME',scope='pool%d'%(i))
                conv_.append(current)
                current = pool

            current=slim.conv2d(pad(current,(fil_s-1)/2),chan_[num_down],[fil_s,fil_s], 
                                weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='VALID',scope='g_conv%d'%(num_down),reuse=reuse)
            contract_temp = current
        ##
        with tf.variable_scope('local_ops',reuse=reuse):
            current = contract_temp
            for ii in range(num_block):
                adding = current
                for j in range(num_conv):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                          weights_initializer=w_init,biases_initializer=b_init, rate=rate[num_down],activation_fn=act,padding='SAME',scope='g_conv_block%d_%d'%(ii,j),reuse=reuse)
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init,activation_fn=None,padding='SAME',scope='g_conv_block%d'%(ii),reuse=reuse)
                if is_residual is True:                
                    current = act(current + adding)
                else:
                    current = act(current) 
            restore_temp = current

        if is_global is True:
            with tf.variable_scope('global_ops',reuse=reuse):
                current = contract_temp
                '''
                for i in range(3):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],activation_fn=act,scope='global%d'%(i),reuse=reuse)
                    current = slim.max_pool2d(current, [2, 2], padding='SAME',scope='global_pool%d'%(i))
                '''
                global_feature = tf.reduce_mean(current,[1,2],keepdims=False)
                current = slim.fully_connected(global_feature,chan_[num_down]*2,activation_fn=lrelu,scope='fully_enhan00',reuse=reuse)
                current = slim.fully_connected(current,chan_[num_down]*2,activation_fn=None,scope='fully_enhan01',reuse=reuse)
                global_feature = tf.reshape(current,[-1,1,1,chan_[num_down]*2])
            restore_temp = act(restore_temp*global_feature[:,:,:,0:chan_[num_down]] + global_feature[:,:,:,chan_[num_down]:])
        
        multis_list = []
        with tf.variable_scope('expanding_ops',reuse=reuse):
            current = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(current,chan_[index_current],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for ii in range(num_block):
                    adding = current
                    for j in range(num_conv):
                        current=slim.conv2d(current, chan_[index_current],[fil_s,fil_s], 
                                            weights_initializer=w_init,biases_initializer=b_init,rate=rate[index_current], padding='SAME',activation_fn=act,scope='g_dconv_block%d%d_%d'%(i,ii,j),reuse=reuse)
                    current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=None,scope='g_dconv_block%d%d'%(i,ii),reuse=reuse)
                    if is_residual is True:
                        current = act(current + adding)
                    else:
                        current = act(current) 
                if i is not (num_down-1):  
                    multis_list.append(slim.conv2d(current, num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=tf.nn.tanh,scope='super%d'%(i),reuse=reuse))    
                
            final = slim.conv2d(current,  num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=tf.nn.tanh,scope='final',reuse=reuse) 

    return final,multis_list




def BilateralNet(input,spatial_bin=128,intensity_bin=8,is_glob_pool=True,net_input_size=512,coef=12,last_chan=96,reuse=False):  
    ## Preprocessing 
    act = lrelu
    with tf.variable_scope('Enhancement',reuse=reuse):
        shape = tf.shape(input)
        if is_glob_pool==True:
            H,W = tf.cast(tf.round(shape[1]/6),tf.int32),tf.cast(tf.round(shape[2]/6),tf.int32)
        else:
            H,W = 512,512
        start = tf.image.resize_images(input,[H,W])

        with tf.variable_scope('splat',reuse=reuse):
            n_ds_layers = int(np.log2(net_input_size/spatial_bin))
            current = start
            for i in range(n_ds_layers):
                chan = 32*(2**(i))
                current = slim.conv2d(current,chan,[3,3], stride=1, activation_fn=act,scope='conv_%d'%(i),reuse=reuse)
                current = slim.conv2d(current,chan,[3,3], stride=2, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            splat_out = current

        with tf.variable_scope('global',reuse=reuse):
            current = splat_out
            for i in range(2):
                current = slim.conv2d(current,64,[3,3], stride=2, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            _, lh, lw, lc = current.get_shape().as_list()
            if is_glob_pool == False:
                current = tf.reshape(current, [-1, lh*lw*lc])  # flattening
            else:
                current = tf.reduce_mean(current,[1,2],keepdims=False)

            current = slim.fully_connected(current,256,normalizer_fn=None,activation_fn=act,scope='fully_rest00',reuse=reuse)
            current = slim.fully_connected(current,128,normalizer_fn=None,activation_fn=act,scope='fully_rest01',reuse=reuse)
            current = slim.fully_connected(current,last_chan,normalizer_fn=None,activation_fn=act,scope='fully_rest02',reuse=reuse)
            current = tf.reshape(current,[-1,1,1,last_chan])
            global_out = current

        with tf.variable_scope('local',reuse=reuse):
            for i in range(2):
                current = slim.conv2d(current,last_chan,[3,3], stride=1, activation_fn=act,scope='conv%d'%(i),reuse=reuse)
            local_out = current

        with tf.variable_scope('fusion',reuse=reuse):
            grid_chan_size = intensity_bin*coef
            current = act(local_out + global_out)
            A = slim.conv2d(current,grid_chan_size,[3,3], stride=1, activation_fn=None,scope='conv',reuse=reuse)

        with tf.variable_scope('guide_curve'):
            npts = 15
            nchans = 3

            idtity = np.identity(nchans, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
            ccm = tf.get_variable('ccm', dtype=tf.float32, initializer=idtity)   # initializer could be np array
            ccm_bias = tf.get_variable('ccm_bias', shape=[nchans,], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            guidemap = tf.matmul(tf.reshape(input, [-1, nchans]), ccm)    #input_tensor shap should be (1,hei,wid,nchans),or will be faulty
            guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')  #bias: A 1-D Tensor with size matching the last dimension of value.
            guidemap = tf.reshape(guidemap, tf.shape(input))

            shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
            shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, np.newaxis,:]
            shifts_ = np.tile(shifts_, (1, 1, 1, nchans, 1))

            guidemap = tf.expand_dims(input, 4)   # 5
            shifts = tf.get_variable('shifts', dtype=tf.float32, initializer=shifts_)

            slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
            slopes_[:, :, :, :, 0] = 1.0
            slopes = tf.get_variable('slopes', dtype=tf.float32, initializer=slopes_)

            guidemap = tf.reduce_sum(slopes*tf.nn.relu(guidemap-shifts), reduction_indices=[4])

            guidemap = slim.conv2d(inputs=guidemap,num_outputs=1, kernel_size=1, weights_initializer=tf.constant_initializer(1.0/nchans),
                                    biases_initializer=tf.constant_initializer(0),activation_fn=None, reuse=reuse,scope='channel_mixing')

            guidemap = tf.clip_by_value(guidemap, 0, 1)

        with tf.variable_scope('guided_upsample'):
            out = []
            input_aug = tf.concat([input,tf.ones_like(input[:,:,:,0:1],dtype=tf.float32)],3)
            shape = tf.shape(A)
            A = tf.reshape(A,[shape[0],shape[1],shape[2],intensity_bin,coef])
            Au = layers.guided_upsampling(A,guidemap)
            for i in range(3):
                out.append(tf.reduce_sum(input_aug*Au[:,:,:,i*4:(i+1)*4],3,keepdims=True))		
            final = tf.concat(out,3)

        
           
    return final





def df_kpn(input_rgbs,noise,filt_s=5,reuse=False):
    def get_pixel_value(img,x,y,z):   # img B,H,W,F,3
        shape = tf.shape(x)  # x,y,z: B,H,W,Sam
        batch_size = shape[0]
        hei,wid,sam = shape[1],shape[2],shape[3]  # B,H,W,Sam
        batch_idx = tf.range(0,batch_size)
        batch_idx = tf.reshape(batch_idx,[batch_size,1,1,1])
        b = tf.tile(batch_idx,[1,hei,wid,sam])
        indices = tf.stack([b,y,x,z],4)
        return tf.gather_nd(img,indices)
        
    input_rgbs = tf.transpose(input_rgbs,[0,1,2,4,3])    # B H W F C
    input_lums = tf.reduce_mean(input_rgbs,4,keepdims=False)    # B H W F 1
    shape = tf.shape(input_lums)
    num_samples = 3*filt_s*filt_s
    ## Offset net
    with tf.variable_scope('Offset_N'):
        offsets = U_net22(input_lums,num_down=3,num_block=1,num_conv=1,num_out=3*num_samples,rate=[1]*10,fil_s=filt_s,is_residual=False,
                         start_chan=32,act=relu,is_global=False,name='Offset_N',reuse=reuse)
        offsets_shape = tf.shape(offsets)
        offsets_r = tf.reshape(offsets,[offsets_shape[0],offsets_shape[1],offsets_shape[2],3,num_samples]) # B,H,W,F,Sam
        
    with tf.variable_scope('Sampler'):
    ## Sampler    
        offsets_x = offsets_r[:,:,:,0,:]
        offsets_y = offsets_r[:,:,:,1,:]
        offsets_z = offsets_r[:,:,:,2,:]
        
        x = tf.linspace(-1.0,1.0,shape[2])
        y = tf.linspace(-1.0,1.0,shape[1])
        
        x,y = tf.meshgrid(x,y)
        x,y = tf.reshape(x,[-1,shape[1],shape[2],1]),tf.reshape(y,[-1,shape[1],shape[2],1])
        x,y = tf.tile(x,[1,1,1,num_samples]),tf.tile(y,[1,1,1,num_samples])
        x_t,y_t = x+offsets_x,y+offsets_y
        z_t = offsets_z
    
        max_x,max_y,max_z = shape[2]-1,shape[1]-1,shape[3]-1 # int
        zero = tf.zeros([],tf.int32)    # int
        x_t_sc = (x_t+1.0)*0.5*tf.cast(max_x,tf.float32)   #float
        y_t_sc = (y_t+1.0)*0.5*tf.cast(max_y,tf.float32)
        z_t_sc = (z_t+1.0)*0.5*tf.cast(max_z,tf.float32)
        
        x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z_t_sc),tf.int32)
        z1 = z0 + 1
        
        x0,x1 = tf.clip_by_value(x0,zero,max_x),tf.clip_by_value(x1,zero,max_x)  #int
        y0,y1 = tf.clip_by_value(y0,zero,max_y),tf.clip_by_value(y1,zero,max_y)
        z0,z1 = tf.clip_by_value(z0,zero,max_z),tf.clip_by_value(z1,zero,max_z)
        
        I000 = get_pixel_value(input_rgbs,x0,y0,z0)   # float
        I001 = get_pixel_value(input_rgbs,x0,y0,z1)
        I010 = get_pixel_value(input_rgbs,x0,y1,z0)
        I011 = get_pixel_value(input_rgbs,x0,y1,z1)
        I100 = get_pixel_value(input_rgbs,x1,y0,z0)
        I101 = get_pixel_value(input_rgbs,x1,y0,z1)
        I110 = get_pixel_value(input_rgbs,x1,y1,z0)
        I111 = get_pixel_value(input_rgbs,x1,y1,z1)
         
        w000 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w001 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w010 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w011 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w100 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w101 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w110 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w111 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        
        w000,w001,w010,w011 = tf.expand_dims(w000,4),tf.expand_dims(w001,4),tf.expand_dims(w010,4),tf.expand_dims(w011,4)
        w100,w101,w110,w111 = tf.expand_dims(w100,4),tf.expand_dims(w101,4),tf.expand_dims(w110,4),tf.expand_dims(w111,4)
        align_out = tf.add_n([w000*I000,w001*I001,w010*I010,w011*I011,w100*I100,w101*I101,w110*I110,w111*I111])
        
    
    ## kpn
    with tf.variable_scope('KPN'):
        shape_input_rgbs = tf.shape(input_rgbs)
        input_rgbs_m = tf.reshape(input_rgbs,[shape_input_rgbs[0],shape_input_rgbs[1],shape_input_rgbs[2],
                                              shape_input_rgbs[3]*shape_input_rgbs[4]])
    
        shape_align_out = tf.shape(align_out)
        align_out_m = tf.reshape(align_out,[shape_align_out[0],shape_align_out[1],shape_align_out[2],
                                              shape_align_out[3]*shape_align_out[4]])            
        input_kpn = tf.concat([input_rgbs_m,noise,offsets,align_out_m],3)    
        kernels = Conv_block(input_kpn,num_conv=3,num_out=num_samples,rate=[1]*10,fil_s=5,chan=64,act=relu,name='KPN',reuse=reuse)
        #kernels = U_net22(input_kpn,num_down=3,num_block=1,num_conv=1,num_out=filt_s*filt_s*3,rate=[1]*10,fil_s=filt_s, 
        #                    is_residual=False,start_chan=32,act=lrelu,is_global=False,name='KPN',reuse=reuse)  
        kernels = tf.expand_dims(kernels,4)
        rgb_filter = kernels*align_out
        axu_out = []
        for i in range(3):
            axu_out.append(3.0*tf.reduce_sum(rgb_filter[:,:,:,i*9:i*9+9,:],3,keepdims=False))
        rgb_out = tf.reduce_sum(rgb_filter,3,keepdims=False)
    return rgb_out,axu_out 
    
    
def df_kpn_enhan(input_rgbs,noise,filt_s=5,reuse=False):
    def get_pixel_value(img,x,y,z):   # img B,H,W,F,3
        shape = tf.shape(x)  # x,y,z: B,H,W,Sam
        batch_size = shape[0]
        hei,wid,sam = shape[1],shape[2],shape[3]  # B,H,W,Sam
        batch_idx = tf.range(0,batch_size)
        batch_idx = tf.reshape(batch_idx,[batch_size,1,1,1])
        b = tf.tile(batch_idx,[1,hei,wid,sam])
        indices = tf.stack([b,y,x,z],4)
        return tf.gather_nd(img,indices)  # B,H,W,Sam,C
        
    input_rgbs = tf.transpose(input_rgbs,[0,1,2,4,3])    # B H W F C
    input_lums = tf.reduce_mean(input_rgbs,4,keepdims=False)    # B H W F 1
    shape = tf.shape(input_lums)
    num_samples = 3*filt_s*filt_s
    ## Offset net
    with tf.variable_scope('Offset_N'):
        w_init = tf.constant_initializer(0.0)
        b_init = tf.constant_initializer(0.0)
        offsets = U_net22(input_lums,num_down=3,num_block=1,num_conv=1,w_init=w_init,b_init=b_init,num_out=3*num_samples,rate=[1]*10,fil_s=filt_s,is_residual=False,
                         start_chan=32,act=lrelu,is_global=False,name='Offset_N',reuse=reuse)
        offsets_shape = tf.shape(offsets)
        offsets_r = tf.reshape(offsets,[offsets_shape[0],offsets_shape[1],offsets_shape[2],3,num_samples]) # B,H,W,F,Sam
        
    with tf.variable_scope('Sampler'):
    ## Sampler    
        offsets_x = offsets_r[:,:,:,0,:]
        offsets_y = offsets_r[:,:,:,1,:]
        offsets_z = offsets_r[:,:,:,2,:]
        
        x = tf.linspace(-1.0,1.0,shape[2])
        y = tf.linspace(-1.0,1.0,shape[1])
        
        x,y = tf.meshgrid(x,y)
        x,y = tf.reshape(x,[-1,shape[1],shape[2],1]),tf.reshape(y,[-1,shape[1],shape[2],1])
        x,y = tf.tile(x,[1,1,1,num_samples]),tf.tile(y,[1,1,1,num_samples])
        x_t,y_t = x+offsets_x,y+offsets_y
        z_t = offsets_z
    
        max_x,max_y,max_z = shape[2]-1,shape[1]-1,shape[3]-1 # int
        zero = tf.zeros([],tf.int32)    # int
        x_t_sc = (x_t+1.0)*0.5*tf.cast(max_x,tf.float32)   #float
        y_t_sc = (y_t+1.0)*0.5*tf.cast(max_y,tf.float32)
        z_t_sc = (z_t+1.0)*0.5*tf.cast(max_z,tf.float32)
        
        x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z_t_sc),tf.int32)
        z1 = z0 + 1
        
        x0,x1 = tf.clip_by_value(x0,zero,max_x),tf.clip_by_value(x1,zero,max_x)  #int
        y0,y1 = tf.clip_by_value(y0,zero,max_y),tf.clip_by_value(y1,zero,max_y)
        z0,z1 = tf.clip_by_value(z0,zero,max_z),tf.clip_by_value(z1,zero,max_z)
        
        I000 = get_pixel_value(input_rgbs,x0,y0,z0)   # float
        I001 = get_pixel_value(input_rgbs,x0,y0,z1)
        I010 = get_pixel_value(input_rgbs,x0,y1,z0)
        I011 = get_pixel_value(input_rgbs,x0,y1,z1)
        I100 = get_pixel_value(input_rgbs,x1,y0,z0)
        I101 = get_pixel_value(input_rgbs,x1,y0,z1)
        I110 = get_pixel_value(input_rgbs,x1,y1,z0)
        I111 = get_pixel_value(input_rgbs,x1,y1,z1)
         
        w000 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w001 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w010 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w011 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w100 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w101 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w110 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w111 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        
        w000,w001,w010,w011 = tf.expand_dims(w000,4),tf.expand_dims(w001,4),tf.expand_dims(w010,4),tf.expand_dims(w011,4)
        w100,w101,w110,w111 = tf.expand_dims(w100,4),tf.expand_dims(w101,4),tf.expand_dims(w110,4),tf.expand_dims(w111,4)
        align_out = tf.add_n([w000*I000,w001*I001,w010*I010,w011*I011,w100*I100,w101*I101,w110*I110,w111*I111])
        
    
    ## kpn
    with tf.variable_scope('KPN'):
        shape_input_rgbs = tf.shape(input_rgbs)
        input_rgbs_m = tf.reshape(input_rgbs,[shape_input_rgbs[0],shape_input_rgbs[1],shape_input_rgbs[2],
                                              shape_input_rgbs[3]*shape_input_rgbs[4]])
    
        shape_align_out = tf.shape(align_out)
        align_out_m = tf.reshape(align_out,[shape_align_out[0],shape_align_out[1],shape_align_out[2],
                                              shape_align_out[3]*shape_align_out[4]])            
        input_kpn = tf.concat([input_rgbs_m,noise,offsets,align_out_m],3)    
        #kernels = Conv_block(input_kpn,num_conv=3,num_out=num_samples,rate=[1]*10,fil_s=5,chan=64,act=relu,name='KPN',reuse=reuse)
        kernels = U_net22(input_kpn,num_down=3,num_block=1,num_conv=1,num_out=filt_s*filt_s*3,rate=[1]*10,fil_s=filt_s, 
                           is_residual=False,start_chan=32,act=lrelu,is_global=True,name='KPN',reuse=reuse)  
        kernels = tf.expand_dims(kernels,4)
        rgb_filter = kernels*align_out
        axu_out = []
        for i in range(3):
            axu_out.append(3.0*tf.reduce_sum(rgb_filter[:,:,:,i*9:i*9+9,:],3,keepdims=False))
        rgb_out = tf.reduce_sum(rgb_filter,3,keepdims=False)
    return rgb_out,axu_out 


def df_kpn_enhan_1(input_rgbs,noise,num_fr=5,filt_s=3,reuse=False):
    ## different anealing loss
    def get_pixel_value(img,x,y):   # img B,H,W,3
        shape = tf.shape(x)  # x,y: B,H,W,1
        batch_size = shape[0]
        hei,wid = shape[1],shape[2] # B,H,W,1
        batch_idx = tf.range(0,batch_size)
        batch_idx = tf.reshape(batch_idx,[batch_size,1,1,1])
        b = tf.tile(batch_idx,[1,hei,wid,1])
        indices = tf.concat([b,y,x],3)
        return tf.gather_nd(img,indices)  # B,H,W,C
        
    input_rgbs = tf.transpose(input_rgbs,[0,1,2,4,3])    # B H W F C
    input_lums = tf.reduce_mean(input_rgbs,4,keepdims=False)    # B H W F 1
    shape = tf.shape(input_lums)
    B,H,W,F = shape[0],shape[1],shape[2],shape[3]
    num_samples = filt_s*filt_s
    ## Offset net
    with tf.variable_scope('Offset_N'):
        w_init = tf.constant_initializer(0.0)
        b_init = tf.constant_initializer(0.0)
        offsets = U_net22(input_lums,num_down=3,num_block=1,num_conv=1,w_init=w_init,b_init=b_init,num_out=2*num_fr,
                          rate=[1]*10,fil_s=filt_s,is_residual=False,start_chan=32,act=lrelu,is_global=False,name='Offset_N',reuse=reuse)
        offsets_shape = tf.shape(offsets)
        offsets_r = tf.reshape(offsets,[offsets_shape[0],offsets_shape[1],offsets_shape[2],2,num_fr]) # B,H,W,2,F
        
    with tf.variable_scope('Sampler'):
        ## generate initail grid with initial filter kernel locations
        x = tf.linspace(0.0,1.0,W)
        y = tf.linspace(0.0,1.0,H)
        x,y = tf.meshgrid(x,y)
        x,y = tf.reshape(x,[-1,H,W,1]),tf.reshape(y,[-1,H,W,1])
        x,y = tf.tile(x,[1,1,1,num_fr]),tf.tile(y,[1,1,1,num_fr])  # B,H,W,F       
        ## adding offsets for each frame
        aligned_list = []
        for i in range(num_fr):
            offsets_x_current = offsets_r[:,:,:,0,i:i+1] # B,H,W,1
            offsets_y_current = offsets_r[:,:,:,1,i:i+1] # B,H,W,1
            x_current,y_current = x[:,:,:,i:i+1],y[:,:,:,i:i+1]
            x_new, y_new = x_current + offsets_x_current, y_current + offsets_y_current
            x_new = tf.clip_by_value(x_new,0.0,1.0)
            y_new = tf.clip_by_value(y_new,0.0,1.0)
            x_t_sc = x_new*(tf.cast(W,tf.float32)-1.0)
            y_t_sc = y_new*(tf.cast(H,tf.float32)-1.0)
       
            x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # # B,H,W,1
            x1 = tf.clip_by_value(x0 + 1,0,W-1)
            y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
            y1 = tf.clip_by_value(y0 + 1,0,H-1)         
            
            I00 = get_pixel_value(input_rgbs[:,:,:,i,:],x0,y0)   # float
            I01 = get_pixel_value(input_rgbs[:,:,:,i,:],x0,y1)
            I10 = get_pixel_value(input_rgbs[:,:,:,i,:],x1,y0)
            I11 = get_pixel_value(input_rgbs[:,:,:,i,:],x1,y1)   # I=B,H,W,C
            
            w00 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)
            w01 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)
            w10 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)
            w11 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)
            
            align_out = tf.add_n([w00*I00,w01*I01,w10*I10,w11*I11])
            aligned_list.append(align_out)
        aligned_imgs = tf.stack(align_out,3)  # B,H,W,F,3  

    
    ## kpn
    with tf.variable_scope('KPN'):
        shape_input_rgbs = tf.shape(input_rgbs)
        input_rgbs_m = tf.reshape(input_rgbs,[shape_input_rgbs[0],shape_input_rgbs[1],shape_input_rgbs[2],
                                              shape_input_rgbs[3]*shape_input_rgbs[4]])
    
        shape_align_out = tf.shape(aligned_imgs)
        align_out_m = tf.reshape(aligned_imgs,[shape_align_out[0],shape_align_out[1],shape_align_out[2],
                                              shape_align_out[3]*shape_align_out[4]])            
        input_kpn = tf.concat([input_rgbs_m,noise,offsets,align_out_m],3)    
        #kernels = Conv_block(input_kpn,num_conv=3,num_out=num_samples,rate=[1]*10,fil_s=5,chan=64,act=relu,name='KPN',reuse=reuse)
        kernels = U_net22(input_kpn,num_down=3,num_block=1,num_conv=1,num_out=filt_s*filt_s*3,rate=[1]*10,fil_s=filt_s, 
                           is_residual=False,start_chan=32,act=lrelu,is_global=True,name='KPN',reuse=reuse)  
        kernels = tf.expand_dims(kernels,4)
        rgb_filter = kernels*align_out
        axu_out = []
        for i in range(3):
            axu_out.append(3.0*tf.reduce_sum(rgb_filter[:,:,:,i*9:i*9+9,:],3,keepdims=False))
        rgb_out = tf.reduce_sum(rgb_filter,3,keepdims=False)
    return rgb_out,axu_out 



def EDVR(input_rgbs,fra_s=5,num_fea=32,reuse = False):
    def dcn(fea,offset,group=8,fil_s=3,name=None,reuse=False):
        def grid_fil(x,y):
            case = [[-1.0,-1.0],[-1.0,0.0],[-1.0,1.0],[0.0,-1.0],[0.0,0.0],[0.0,1.0],[1.0,-1.0],[1.0,0.0],[1.0,1.0]]
            offset_x,offset_y = [],[]
            shape = tf.shape(x)
            for i in case:
                case_x,case_y = 2.0*i[0]/tf.cast(shape[2],tf.float32),2.0*i[1]/tf.cast(shape[1],tf.float32)
                offset_x.append(tf.reshape(case_x,[1,1,1,1]))
                offset_y.append(tf.reshape(case_y,[1,1,1,1]))
            offset_x_,offset_y_ = tf.concat(offset_x,3),tf.concat(offset_y,3)
            x_new,y_new = x+offset_x_,y+offset_y_
            
            return x_new,y_new
        def get_pixel_value(img,x,y):   # img B,H,W,8
            shape = tf.shape(x)  # x,y,z: B,H,W,9
            batch_size = shape[0]
            hei,wid,sam = shape[1],shape[2],shape[3]  # B,H,W,Sam
            batch_idx = tf.range(0,batch_size)
            batch_idx = tf.reshape(batch_idx,[batch_size,1,1,1])
            b = tf.tile(batch_idx,[1,hei,wid,sam])
            indices = tf.stack([b,y,x],4)
            return tf.gather_nd(img,indices)
                
        with tf.variable_scope(name,reuse=reuse):
            offset = Conv_block1(offset,num_conv=1,num_out=group*2*fil_s*fil_s,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='ali',reuse=reuse)
            offset_x,offset_y = offset[:,:,:,0:group*fil_s*fil_s],offset[:,:,:,group*fil_s*fil_s:group*fil_s*fil_s*2]
            #mask = tf.sigmoid(mask)
            shape = tf.shape(offset_x)
            collect = []
            for i in range(group):
                num_c = int(num_fea/group)
                #mask_current = tf.expand_dims(mask[:,:,:,i*fil_s*fil_s:(i+1)*fil_s*fil_s],4)
                fea_current = fea[:,:,:,i*num_c:(i+1)*num_c]
            
                x = tf.linspace(-1.0,1.0,shape[2])
                y = tf.linspace(-1.0,1.0,shape[1])
                
                x,y = tf.meshgrid(x,y)
                x,y = tf.reshape(x,[-1,shape[1],shape[2],1]),tf.reshape(y,[-1,shape[1],shape[2],1])
                x,y = tf.tile(x,[1,1,1,fil_s*fil_s]),tf.tile(y,[1,1,1,fil_s*fil_s])                
                x,y = grid_fil(x,y)   
                x_t,y_t = x+offset_x[:,:,:,i*fil_s*fil_s:(i+1)*fil_s*fil_s],y+offset_y[:,:,:,i*fil_s*fil_s:(i+1)*fil_s*fil_s]
            
                max_x,max_y = shape[2]-1,shape[1]-1 # int
                zero = tf.zeros([],tf.int32)    # int
                x_t_sc = (x_t+1.0)*0.5*tf.cast(max_x,tf.float32)   #float
                y_t_sc = (y_t+1.0)*0.5*tf.cast(max_y,tf.float32)
    
                x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
                x1 = x0 + 1
                y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
                y1 = y0 + 1
                
                x0,x1 = tf.clip_by_value(x0,zero,max_x+1),tf.clip_by_value(x1,zero,max_x+1)  #int
                y0,y1 = tf.clip_by_value(y0,zero,max_y+1),tf.clip_by_value(y1,zero,max_y+1)
                
                padd = [[0,0],[0,1],[0,1],[0,0]]
                I00 = get_pixel_value(tf.pad(fea_current,padd,'SYMMETRIC'),x0,y0)   # float
                I01 = get_pixel_value(tf.pad(fea_current,padd,'SYMMETRIC'),x0,y1)
                I10 = get_pixel_value(tf.pad(fea_current,padd,'SYMMETRIC'),x1,y0)
                I11 = get_pixel_value(tf.pad(fea_current,padd,'SYMMETRIC'),x1,y1)
    
                 
                w00 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)
                w01 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)
                w10 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)
                w11 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)
                
                w00,w01,w10,w11 = tf.expand_dims(w00,4),tf.expand_dims(w01,4),tf.expand_dims(w10,4),tf.expand_dims(w11,4)
                align_out = tf.add_n([w00*I00,w01*I01,w10*I10,w11*I11])
    
                align_out = align_out#*mask_current
                align_out = tf.reshape(align_out,[shape[0],shape[1],shape[2],-1])  # B,H,W,8*9
                collect.append(align_out)
            collect_ = tf.concat(collect,3)
             
            out = Conv_block1(collect_,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=1,chan=num_fea,act=lrelu,name='out',reuse=reuse)
        return out
        
    def PCD_align(nalign,ref,name,reuse=False):   # nalign is B,H,W,C
        with tf.variable_scope(name,reuse=reuse):
            group = 4
            L3_offset = tf.concat([nalign[2],ref[2]],3)
            L3_offset = Conv_block1(L3_offset,num_conv=2,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea3',reuse=reuse)
            L3_fea = dcn(nalign[2],L3_offset,group=group,fil_s=3,name='dcn3',reuse=reuse)
            
            L2_offset = tf.concat([nalign[1],ref[1]],3)
            L2_offset = Conv_block1(L2_offset,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea2',reuse=reuse)*2.0
            L2_offset = upsample_and_concat_c(L3_offset,L2_offset,num_fea, num_fea, 'ou_and_c3',reuse=reuse)
            L2_offset = Conv_block1(L2_offset,num_conv=2,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea21',reuse=reuse)
            L2_fea = dcn(nalign[1],L2_offset,group=group,fil_s=3,name='dcn2',reuse=reuse)
            L2_fea = upsample_and_concat_c(L3_fea,L2_fea,num_fea, num_fea, 'fu_and_c3',reuse=reuse)
            L2_fea = Conv_block1(L2_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea22',reuse=reuse)
            
            L1_offset = tf.concat([nalign[0],ref[0]],3)
            L1_offset = Conv_block1(L1_offset,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea1',reuse=reuse)*2.0
            L1_offset = upsample_and_concat_c(L2_offset,L1_offset,num_fea, num_fea, 'ou_and_c2',reuse=reuse)
            L1_offset = Conv_block1(L1_offset,num_conv=2,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea11',reuse=reuse)
            L1_fea = dcn(nalign[0],L1_offset,group=group,fil_s=3,name='dcn1',reuse=reuse)
            L1_fea = upsample_and_concat_c(L2_fea,L1_fea,num_fea, num_fea, 'fu_and_c2',reuse=reuse)
            L1_fea = Conv_block1(L1_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea12',reuse=reuse)
            
            offset = tf.concat([L1_fea,ref[0]],3)
            offset = Conv_block1(offset,num_conv=2,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea0',reuse=reuse)
            L1_fea = dcn(L1_fea,offset,group=group,fil_s=3,name='dcn0',reuse=reuse)
        return L1_fea     # B H W 64
        
    input_rgbs = tf.transpose(input_rgbs,[0,4,1,2,3])    # B F H W  C
    in_shape = tf.shape(input_rgbs)   # B F H W  C
    B, F, H, W, C = in_shape[0],in_shape[1],in_shape[2],in_shape[3],in_shape[4]
    center = int((fra_s+1)/2-1)
    with tf.variable_scope('Fea_extr'):
        current = input_rgbs   # B F,H W  C
        current = tf.reshape(input_rgbs,[-1, H, W, C])
        current = Conv_block1(current,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='pre_fea0',reuse=reuse)
        current = Conv_block_residual(current,num_block=2,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,
                                      name='pre_fea1',reuse=reuse)
        Fea_extr = current
    with tf.variable_scope('Align'):
        current = Fea_extr
        L1_fea = Conv_block1(current,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea00',reuse=reuse)
        L1_fea = Conv_block1(L1_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea01',reuse=reuse)    
        
        L2_fea = Conv_block1(L1_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea10',reuse=reuse)
        L2_fea = Conv_block1(L2_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea11',reuse=reuse)
        L2_fea=slim.conv2d(L2_fea,num_fea,[3,3], stride=2, activation_fn=lrelu,padding='SAME',scope='fea13',reuse=reuse)
        #L2_fea=slim.max_pool2d(L2_fea, [3, 3], padding='SAME',scope='fea13')
        
        L3_fea = Conv_block1(L2_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea20',reuse=reuse)
        L3_fea = Conv_block1(L3_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fea21',reuse=reuse)
        L3_fea = slim.conv2d(L3_fea,num_fea,[3,3], stride=2, activation_fn=lrelu,padding='SAME',scope='fea22',reuse=reuse)
        #L3_fea=slim.max_pool2d(L3_fea, [3, 3], padding='SAME',scope='fea22')
    
        L1_fea = tf.reshape(L1_fea,[B,F, H, W, -1])
        
        L2_fea = tf.reshape(L2_fea,[B,F, H//2, W//2, -1])
        L3_fea = tf.reshape(L3_fea,[B,F, H//4, W//4, -1])
        
        L1_center,L2_center,L3_center = L1_fea[:,center:center+1,:,:,:],L2_fea[:,center:center+1,:,:,:],L3_fea[:,center:center+1,:,:,:]  ## center:B,H,W,64
        L1_center,L2_center,L3_center = tf.tile(L1_center,[1,F,1,1,1]),tf.tile(L2_center,[1,F,1,1,1]),tf.tile(L3_center,[1,F,1,1,1])
        L1_center = tf.reshape(L1_center,[B*F,H,W,-1])
        L2_center = tf.reshape(L2_center,[B*F,H//2,W//2,-1])
        L3_center = tf.reshape(L3_center,[B*F,H//4,W//4,-1])
        L1_fea = tf.reshape(L1_fea,[B*F,H,W,-1])
        L2_fea = tf.reshape(L2_fea,[B*F,H//2,W//2,-1])
        L3_fea = tf.reshape(L3_fea,[B*F,H//4,W//4,-1])
        
        nalign,refs = [L1_fea,L2_fea,L3_fea],[L1_center,L2_center,L3_center]
    
        PCD_out_ = PCD_align(nalign,refs,'pcd',reuse=reuse)
        #print(PCD_out_.shape)
        #PCD_out_ = Conv_block1(PCD_out_,num_conv=1,num_out=3,rate=[1]*10,fil_s=1,chan=32,act=relu,name='fuse',reuse=reuse)
        PCD_out_ = tf.reshape(PCD_out_,[B,F,H,W,num_fea])
        #PCD_out_ = tf.transpose(PCD_out_,[0,2,3,4,1])
    
    with tf.variable_scope('TSA'):    
        aligned_fea = PCD_out_
        emb_ref = Conv_block1(aligned_fea[:,center,:,:,:],num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=64,act=lrelu,name='emb_ref',reuse=reuse)
        emb_in = tf.reshape(aligned_fea,[-1,H, W,num_fea])
        emb = Conv_block1(emb_in,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='emb',reuse=reuse)
        emb = tf.reshape(emb,[B,F,H,W,num_fea])
        col_l = []
        for i in range(fra_s):
            emb_cur = emb[:,i,:,:,:]
            color_temp = tf.reduce_sum(emb_cur*emb_ref,3,keepdims=False) # B,H,W
            col_l.append(color_temp)
        col_prob = tf.sigmoid(tf.stack(col_l,1))  # B,N,H,W
        col_prob = tf.tile(tf.expand_dims(col_prob,4),[1,1,1,1,num_fea])
        aligned_fea = aligned_fea*col_prob # B,N,H,W,64
        
        aligned_fea = tf.reshape(tf.transpose(aligned_fea,[0,2,3,1,4]),[B,H,W,-1])
        tfuse_fea = Conv_block1(aligned_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='fuse',reuse=reuse) # B,H,W,64
        # Spatial attention
        att = Conv_block1(aligned_fea,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa0',reuse=reuse)
        add0 = att
        att_max = slim.max_pool2d(att, [3, 3], padding='SAME',scope='spa0max')
        att_aver = slim.avg_pool2d(att, [3, 3], padding='SAME',scope='spa0aver')
        con = tf.concat([att_max,att_aver],3)
        att = Conv_block1(con,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa0a',reuse=reuse) # B,H,W,64
        add1 = att
        
        att = Conv_block1(att,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa1',reuse=reuse) # B,H,W,64
        att_max = slim.max_pool2d(att, [3, 3], padding='SAME',scope='spa1max')
        att_aver = slim.avg_pool2d(att, [3, 3], padding='SAME',scope='spa1aver')
        con = tf.concat([att_max,att_aver],3)
        att = Conv_block1(con,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa1a',reuse=reuse) # B,H,W,64  
        att = Conv_block1(att,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa1af',reuse=reuse) # B,H,W,64 
        
        att = upsample_to(att,add1,num_fea,num_fea,scope='uac0',reuse=reuse)
        att = Conv_block1(att,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa2af',reuse=reuse) # B,H,W,64 
        att = att + add1
        att = Conv_block1(att,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa3af',reuse=reuse) # B,H,W,64 
        att = upsample_to(att,add0,num_fea,num_fea,scope='uac1',reuse=reuse)
        att_add = Conv_block1(att,num_conv=1,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa4af',reuse=reuse) # B,H,W,64 
        att_mul = Conv_block1(att_add,num_conv=2,num_out=num_fea,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='spa5af',reuse=reuse) # B,H,W,64 
        att_mul = tf.sigmoid(att_mul)
        
        fea = tfuse_fea*att_mul*2.0+att_add
        fea = Conv_block_residual(fea,num_block=4,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='out_residual',reuse=reuse)
        out = Conv_block1(fea,num_conv=2,num_out=3,rate=[1]*10,fil_s=3,chan=num_fea,act=lrelu,name='out_conv',reuse=reuse) # B,H,W,64
    
    return out
        

def pack_fea(x,num_fr=5): #B,F,H,W,C
    shape = tf.shape(x)
    B,F,H,W,C = shape[0],shape[1],shape[2],shape[3],shape[4]
    center = int((num_fr+1)/2)
    ref = x[:,center-1:center,:,:,:] #B,1,H,W,C

    inds = np.concatenate([np.arange(0,center-1),np.arange(center,num_fr)],0)
    lis = [x[:,i:i+1,:,:,:] for i in inds]
    others = tf.concat(lis,1) #B,F-1,H,W,C
    
    ref_tile = tf.tile(ref,[1,num_fr-1,1,1,1]) #B,F-1,H,W,C
    ref_tile_r = tf.reshape(ref_tile,[B*(F-1),H,W,C])  #B*(F-1),H,W,C
    others_r = tf.reshape(others,[B*(F-1),H,W,C])      #B*(F-1),H,W,C
    in_feas =tf.concat([others_r,ref_tile_r],3)
    return in_feas,ref,others  #B*(F-1),H,W,C*2, #B,1,H,W,C #B,F-1,H,W,C
    
def get_pixel_value(img,x,y):   # img B,H,W,C
    shape = tf.shape(x)  # x,y,z: B,H,W
    batch_size = shape[0]
    hei,wid = shape[1],shape[2]  # B,H,W
    batch_idx = tf.range(0,batch_size)
    batch_idx = tf.reshape(batch_idx,[batch_size,1,1])
    b = tf.tile(batch_idx,[1,hei,wid])
    indices = tf.stack([b,y,x],3)
    return tf.gather_nd(img,indices)     # B,H,W,C

def get_pixel_value_3D(img,x,y,z):   # img B,H,W,F,C
    shape = tf.shape(x)  # x,y,z: B,H,W
    batch_size = shape[0]
    hei,wid = shape[1],shape[2]  # B,H,W
    batch_idx = tf.range(0,batch_size)
    batch_idx = tf.reshape(batch_idx,[batch_size,1,1])
    b = tf.tile(batch_idx,[1,hei,wid])
    indices = tf.stack([b,y,x,z],3)
    return tf.gather_nd(img,indices)     # B,H,W,C


def U_Net_align_spy(img,num_fr=5,num_down=4,reuse = False):  # img:B,H,W,C,F  noise:B,H,W,1
    def image_warp(images, flow, name='image_warp'):
        with tf.name_scope(name):        
            shape = tf.shape(images)
            batch_size = shape[0]
            height = shape[1]
            width = shape[2]
            channels = shape[3]
            #images = tf.reshape(images,[-1,height,width,channels])
            #flow = tf.reshape(flow,[-1,height,width,2])
    
            x = tf.linspace(0.0,1.0,width)
            y = tf.linspace(0.0,1.0,height)
            grid_x, grid_y = tf.meshgrid(x, y)
            grid_x, grid_y = tf.cast(grid_x,flow.dtype),tf.cast(grid_y,flow.dtype)
            grid_x, grid_y = tf.expand_dims(tf.expand_dims(grid_x,0),3),tf.expand_dims(tf.expand_dims(grid_y,0),3)
            grid_y = (grid_y + flow[:,:,:,0:1])*tf.cast(height-1,flow.dtype)
            grid_x = (grid_x + flow[:,:,:,1:2])*tf.cast(width-1,flow.dtype)
            
            
            grid = tf.concat([grid_y, grid_x], 3) # B,H,W,2
            coords = tf.reshape(grid,[batch_size, height * width, 2]) # B,H*W,2
            coords = tf.stack([tf.minimum(tf.maximum(0.0, coords[:, :, 0]), tf.cast(height, flow.dtype) - 1.0),
                               tf.minimum(tf.maximum(0.0, coords[:, :, 1]), tf.cast(width, flow.dtype) - 1.0)], axis=2)
    
            floors = tf.cast(tf.floor(coords), tf.int32)
            ceils = floors + 1       ## the ceils and floors are not clipped
            alphas = tf.cast(coords - tf.cast(floors, flow.dtype), images.dtype)
            alphas = tf.reshape(tf.minimum(tf.maximum(0.0, alphas), 1.0), shape=[batch_size, height, width, 1, 2])
    
            images_flattened = tf.reshape(images, [-1, channels])
            batch_offsets = tf.expand_dims(tf.range(batch_size) * height * width, axis=1)
    
            def gather(y_coords, x_coords):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(images_flattened, linear_coordinates)
                return tf.reshape(gathered_values, shape)
    
            top_left = gather(floors[:, :, 0], floors[:, :, 1])    # B,H,W,C
            top_right = gather(floors[:, :, 0], ceils[:, :, 1])
            bottom_left = gather(ceils[:, :, 0], floors[:, :, 1])
            bottom_right = gather(ceils[:, :, 0], ceils[:, :, 1])
    
            interp_top = alphas[:, :, :, :, 1] * (top_right - top_left) + top_left
            interp_bottom = alphas[:, :, :, :, 1] * (bottom_right - bottom_left) + bottom_left
            interpolated = alphas[:, :, :, :, 0] * (interp_bottom - interp_top) + interp_top

        return interpolated
    def pack_fea(x,num_fr=5): #B,F,H,W,C
        shape = tf.shape(x)
        B,F,H,W,C = shape[0],shape[1],shape[2],shape[3],shape[4]
        center = int((num_fr+1)/2)
        ref = x[:,center-1:center,:,:,:] #B,1,H,W,C
    
        inds = np.concatenate([np.arange(0,center-1),np.arange(center,num_fr)],0)
        lis = [x[:,i:i+1,:,:,:] for i in inds]
        others = tf.concat(lis,1) #B,F-1,H,W,C
        
        ref_tile = tf.tile(ref,[1,num_fr-1,1,1,1]) #B,F-1,H,W,C
        ref_tile_r = tf.reshape(ref_tile,[B*(F-1),H,W,C])  #B*(F-1),H,W,C
        others_r = tf.reshape(others,[B*(F-1),H,W,C])      #B*(F-1),H,W,C
        in_feas =tf.concat([others_r,ref_tile_r],3)
        return in_feas,ref,others  #B*(F-1),H,W,C*2, #B,1,H,W,C #B,F-1,H,W,C
    def U_net2222(inputlist,num_down=4,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,start_chan=32,act=lrelu,name=None,reuse=False):
        ## parameters
        conv_ = []
        chan_ = []

        if w_init is None:
            w_init = tf.contrib.slim.xavier_initializer()
        if b_init is None:
            b_init = tf.constant_initializer(value=0.0)    
        for i in range(num_down+1):
            chan_.append(start_chan*(2**(i)))
            ##
        multis_list = []
        with tf.variable_scope('local_ops',reuse=reuse):
            current = inputlist[num_down]
            for j in range(num_conv):
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, rate=1,activation_fn=act,padding='SAME',scope='g_conv_block_%d'%(j),reuse=reuse)
            fine_flow = slim.conv2d(current,2,[fil_s,fil_s],
                                  weights_initializer=w_init,biases_initializer=b_init,activation_fn=tf.nn.tanh,padding='SAME',scope='g_conv_block',reuse=reuse)
            
            current_imgs = inputlist[num_down]
            other_img,ref_img = current_imgs[:,:,:,0:3],current_imgs[:,:,:,3:] 
            multis_list.append(image_warp(other_img,fine_flow,'fine_wapring%d'%(num_down)))  
            restore_temp = fine_flow

        
        with tf.variable_scope('expanding_ops',reuse=reuse):
            init_flow = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                current_imgs = inputlist[index_current]
                other_img,ref_img = current_imgs[:,:,:,0:3],current_imgs[:,:,:,3:]                                
                up_flow = slim.conv2d_transpose(init_flow,2,3,(2,2),padding='SAME',scope='up%d'%(i),reuse=reuse)

                other_img_warped = image_warp(other_img,up_flow,'init_wapring%d'%(i))
                fea = tf.concat([other_img_warped,ref_img,up_flow],3)
            
                #current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(fea,chan_[index_current],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for j in range(num_conv):
                    current=slim.conv2d(current, chan_[index_current],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init,rate=1, padding='SAME',activation_fn=act,scope='g_dconv_block%d_%d'%(i,j),reuse=reuse)
                fine_flow = up_flow + slim.conv2d(current,2,[fil_s,fil_s], 
                                    weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=tf.nn.tanh,scope='g_dconv_block%d'%(i),reuse=reuse)
 
                multis_list.append(image_warp(other_img,fine_flow,'fine_wapring%d'%(i)))    
                init_flow = fine_flow
             
    
        return multis_list  
    
        
    img = tf.transpose(img,[0,4,1,2,3])
    shape = tf.shape(img)
    B,F,H,W,C = shape[0],shape[1],shape[2],shape[3],shape[4]

    with tf.variable_scope('Stage1',reuse=reuse):
        current = img
        in_imgs_dn,ref_img_dn,other_imgs_dn = pack_fea(current,num_fr=num_fr)  #B*(F-1),H,W,C*2, #B,1,H,W,C #B,F-1,H,W,C
        in_imgs_dn = tf.clip_by_value(in_imgs_dn,0.0,1.0)
        w_init = tf.constant_initializer(0.0)
        b_init = tf.constant_initializer(0.0)
        rate = [1,2,2,4,4,8]
        inlist = []
        for i in range(num_down+1):
            size = [tf.cast(H/(2**i),tf.int32),tf.cast(W/(2**i),tf.int32)]
            
            inlist.append(tf.image.resize_bilinear(in_imgs_dn,size))

        warped_imgs_list = U_net2222(inlist,num_down=num_down,num_conv=6,num_out=2,rate=rate,fil_s=5,w_init=None,b_init=None,
                                start_chan=32,act=lrelu,name='Alignment',reuse=reuse)  # B*(F-1),H,W,2 
        
        for i in range(len(warped_imgs_list)):
            current = warped_imgs_list[i]
            shape = tf.shape(current)
            H_c,W_c,C_c = shape[1],shape[2],shape[3]
            warped_imgs_list[i] = tf.transpose(tf.reshape(current,[B,F-1,H_c,W_c,C_c]),[0,2,3,4,1])
        warped_imgs_list.reverse()      
    return warped_imgs_list


  
def LiangNN(img,noise,num_fr=5,reuse = False):  # img:B,H,W,C,F  noise:B,H,W,1
    img = tf.transpose(img,[0,4,1,2,3])
    center = int((num_fr+1)/2-1)
    shape = tf.shape(img)
    B,F,H,W,C = shape[0],shape[1],shape[2],shape[3],shape[4]
    
    ##  Single frame denoising
    with tf.variable_scope('Stage0',reuse=reuse):
        current = tf.reshape(img,[B*F,H,W,C])
        noise_exp = tf.tile(noise,[B*F,1,1,1])
        current = tf.concat([current,noise_exp],3)
        single_dn_out = U_net22(current,num_down=3,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,is_residual=False,
                                start_chan=32,act=lrelu,is_global=False,name='SingleDN',reuse=reuse)
        single_dn_out = tf.reshape(single_dn_out,[B,F,H,W,C])  # B,F,H,W,C
        ref_img_dn = single_dn_out[:,center:center+1,:,:,:]
        single_dn_out_final = tf.transpose(single_dn_out,[0,2,3,4,1])
    ### Alignment
    
    current = (tf.clip_by_value(single_dn_out_final,0.0,1.0)+1e-4)**(1.0/2.2)
    warped_imgs_list = U_Net_align_spy(current,num_fr=num_fr,num_down=4,reuse=reuse)
    warped_img = warped_imgs_list[0]
    warped_img_out = (tf.clip_by_value(warped_img,0.0,1.0)+1e-4)**2.2

    
    warped_img = tf.transpose(warped_img_out,[0,4,1,2,3])
    
    ### Fusion
    with tf.variable_scope('Stage2',reuse=reuse):
        chan = 32
        bs = 15
        #current0 = tf.concat([align_out[:,0:center,:,:,:],ref_img,align_out[:,center:,:,:,:]],1)  # B,F,H,W,C
        current = tf.concat([warped_img[:,0:center,:,:,:],ref_img_dn,warped_img[:,center:,:,:,:]],1)  # B,F,H,W,C
        #current = tf.concat([current0,current1],4)  # B,F,H,W,C*2
        #current = tf.concat([current,tf.tile(ref_img_dn,[1,F,1,1,1])],4)  # B,F,H,W,C*3
        current = tf.reshape(current,[B*F,H,W,C])   # B*F,H,W,C*2
        #current = tf.concat([current,noise_exp],3)  # B*F,H,W,C*3+1
        feas = Conv_block_residual(current,num_block=4,rate=[1]*10,fil_s=3,chan=chan,act=lrelu,name='out_residual',reuse=reuse)
        feas = tf.reshape(feas,[B,F,H,W,chan])
        ref_fea = feas[:,center,:,:,:]  # B,H,W,chan
        other_feas = tf.concat([feas[:,:center,:,:,:],feas[:,center+1:,:,:,:]],1)  # B,F-1,H,W,chan
        alphas = tf.Variable(np.ones(shape=[1,1,1,chan],dtype=np.float32),dtype=tf.float32)
        out_list1 = []
        for i in range(num_fr-1):
            other_fea_cur = other_feas[:,i,:,:,:]
            ref_fea_p = pad(ref_fea,p=(bs-1)/2)
            other_fea_cur_p = pad(other_fea_cur,p=(bs-1)/2)
            residual = tf.abs(ref_fea_p-other_fea_cur_p)
            weight = slim.separable_conv2d(residual,num_outputs=None,kernel_size=bs,padding='VALID',
                                  weights_initializer=tf.constant_initializer(value=1.0/(bs*bs)),
                                  biases_initializer=None,scope='we%d'%(i),reuse=reuse)
            weight = tf.exp(-alphas*weight)
            out_list1.append(weight*other_fea_cur+(1.0-weight)*ref_fea)
        
        out = tf.reduce_mean(tf.stack(out_list1,1),1)   
        out_final = Conv_block1(out,num_conv=4,num_out=3,rate=[1]*10,fil_s=1,chan=3,act=lrelu,name='out_conv',reuse=reuse) # B,H,W,3     
    
    return single_dn_out_final,warped_img_out,out_final


    
def U_Net_align(img,num_fr=5,reuse = False):  # img:B,H,W,C,F  noise:B,H,W,1
    def U_net2222(input,num_down=4,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,start_chan=32,act=lrelu,name=None,reuse=False):
        ## parameters
        conv_ = []
        chan_ = []
        if w_init is None:
            w_init = tf.contrib.slim.xavier_initializer()
        if b_init is None:
            b_init = tf.constant_initializer(value=0.0)    
        for i in range(num_down+1):
            chan_.append(start_chan*(2**(i)))
        
        with tf.variable_scope(name,reuse=reuse):
            current = input
            with tf.variable_scope('contracting_ops',reuse=reuse):
                for i in range(num_down):
                    current = slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                          weights_initializer=w_init,biases_initializer=b_init,
                                          activation_fn=act,scope='g_conv%d'%(i),padding='SAME',reuse=reuse)
                    for j in range(num_conv):
                        current=slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                                 weights_initializer=w_init,biases_initializer=b_init,rate=rate[i], activation_fn=act,padding='SAME',scope='g_conv%d_block_%d'%(i,j),reuse=reuse)
                    current=slim.conv2d(current,chan_[i],[fil_s,fil_s], 
                                            weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='SAME',scope='g_conv%d_block'%(i),reuse=reuse)
                    pool=slim.conv2d(current,chan_[i],[fil_s,fil_s], stride=2,
                                            weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='SAME',scope='pool%d'%(i),reuse=reuse)        
                    #pool=slim.max_pool2d(current, [2, 2], padding='SAME',scope='pool%d'%(i))
                    conv_.append(current)
                    current = pool
    
                current=slim.conv2d(pad(current,(fil_s-1)/2),chan_[num_down],[fil_s,fil_s], 
                                    weights_initializer=w_init,biases_initializer=b_init, activation_fn=act,padding='VALID',scope='g_conv%d'%(num_down),reuse=reuse)
                contract_temp = current
            
            ##
            with tf.variable_scope('local_ops',reuse=reuse):
                current = contract_temp
                for j in range(num_conv):
                    current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                          weights_initializer=w_init,biases_initializer=b_init, rate=rate[num_down],activation_fn=act,padding='SAME',scope='g_conv_block_%d'%(j),reuse=reuse)
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init,activation_fn=act,padding='SAME',scope='g_conv_block',reuse=reuse)
                
                restore_temp = current
            
            multis_list = []
            with tf.variable_scope('expanding_ops',reuse=reuse):
                current = restore_temp
                for i in range(num_down):
                    index_current = num_down-1-i
                    current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                    current = slim.conv2d(current,chan_[index_current],[fil_s,fil_s],
                                          weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                    for j in range(num_conv):
                        current=slim.conv2d(current, chan_[index_current],[fil_s,fil_s], 
                                            weights_initializer=w_init,biases_initializer=b_init,rate=rate[index_current], padding='SAME',activation_fn=act,scope='g_dconv_block%d_%d'%(i,j),reuse=reuse)
                    current=slim.conv2d(current,  chan_[index_current],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=tf.nn.tanh,scope='g_dconv_block%d'%(i),reuse=reuse)
    
                    if i is not (num_down-1):  
                        multis_list.append(slim.conv2d(current, num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=tf.nn.tanh,scope='super%d'%(i),reuse=reuse))    
                    
                final = slim.conv2d(current,  num_out,[1,1], weights_initializer=w_init,biases_initializer=b_init, activation_fn=tf.nn.tanh,scope='final',reuse=reuse) 
    
        return final,multis_list  
    def image_warp(images, flow, name='image_warp'):
        with tf.name_scope(name):        
            shape = tf.shape(images)
            batch_size = shape[1]
            height = shape[2]
            width = shape[3]
            frame_s = shape[1]
            channels = shape[4]
            images = tf.reshape(images,[-1,height,width,channels])
            flow = tf.reshape(flow,[-1,height,width,2])
    
            x = tf.linspace(0.0,1.0,width)
            y = tf.linspace(0.0,1.0,height)
            grid_x, grid_y = tf.meshgrid(x, y)
            grid_x, grid_y = tf.cast(grid_x,flow.dtype),tf.cast(grid_y,flow.dtype)
            grid_x, grid_y = tf.expand_dims(tf.expand_dims(grid_x,0),3),tf.expand_dims(tf.expand_dims(grid_y,0),3)
            grid_y = (grid_y + flow[:,:,:,0:1])*tf.cast(height-1,flow.dtype)
            grid_x = (grid_x + flow[:,:,:,1:2])*tf.cast(width-1,flow.dtype)
            
            
            grid = tf.concat([grid_y, grid_x], 3) # B,H,W,2
            coords = tf.reshape(grid,[batch_size, height * width, 2]) # B,H*W,2
            coords = tf.stack([tf.minimum(tf.maximum(0.0, coords[:, :, 0]), tf.cast(height, flow.dtype) - 1.0),
                               tf.minimum(tf.maximum(0.0, coords[:, :, 1]), tf.cast(width, flow.dtype) - 1.0)], axis=2)
    
            floors = tf.cast(tf.floor(coords), tf.int32)
            ceils = floors + 1       ## the ceils and floors are not clipped
            alphas = tf.cast(coords - tf.cast(floors, flow.dtype), images.dtype)
            alphas = tf.reshape(tf.minimum(tf.maximum(0.0, alphas), 1.0), shape=[batch_size, height, width, 1, 2])
    
            images_flattened = tf.reshape(images, [-1, channels])
            batch_offsets = tf.expand_dims(tf.range(batch_size) * height * width, axis=1)
    
            def gather(y_coords, x_coords):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(images_flattened, linear_coordinates)
                return tf.reshape(gathered_values, shape)
    
            top_left = gather(floors[:, :, 0], floors[:, :, 1])    # B,H,W,C
            top_right = gather(floors[:, :, 0], ceils[:, :, 1])
            bottom_left = gather(ceils[:, :, 0], floors[:, :, 1])
            bottom_right = gather(ceils[:, :, 0], ceils[:, :, 1])
    
            interp_top = alphas[:, :, :, :, 1] * (top_right - top_left) + top_left
            interp_bottom = alphas[:, :, :, :, 1] * (bottom_right - bottom_left) + bottom_left
            interpolated = alphas[:, :, :, :, 0] * (interp_bottom - interp_top) + interp_top
    
            interpolated = tf.reshape(interpolated, [1,frame_s,height,width,channels])   # this should be right
            interpolated = tf.transpose(interpolated,[0,2,3,4,1])  # B,H,W,C,F-1
        return interpolated
        
    img = tf.transpose(img,[0,4,1,2,3])
    shape = tf.shape(img)
    B,F,H,W,C = shape[0],shape[1],shape[2],shape[3],shape[4]

    with tf.variable_scope('Stage1',reuse=reuse):
        current = img
        in_imgs_dn,ref_img_dn,other_imgs_dn = pack_fea(current,num_fr=num_fr)  #B*(F-1),H,W,C*2, #B,1,H,W,C #B,F-1,H,W,C
        in_imgs_dn = tf.clip_by_value(in_imgs_dn,0.0,1.0)
        w_init = tf.constant_initializer(0.0)
        b_init = tf.constant_initializer(0.0)
        rate = [1,2,2,4,4,8]
        offsets, offsets_list = U_net2222(in_imgs_dn,num_down=4,num_conv=1,num_out=2,rate=rate,fil_s=3,w_init=None,b_init=None,
                                start_chan=32,act=lrelu,name='Alignment',reuse=reuse)  # B*(F-1),H,W,2
        offsets = tf.reshape(offsets,[B,F-1,H,W,2])
        other_imgs_dn_suplist = []
        for i in range(len(offsets_list)):
            current = offsets_list[i]  # B*(F-1),?,?,2
            current_shape = tf.shape(current)
            H_,W_ = current_shape[1],current_shape[2]
            current_other_imgs_dn = tf.image.resize_bilinear(tf.reshape(other_imgs_dn,[B*(F-1),H,W,C]),[H_,W_])
            current_other_imgs_dn = tf.reshape(current_other_imgs_dn,[B,F-1,H_,W_,C])
            other_imgs_dn_suplist.append(current_other_imgs_dn)
            offsets_list[i] = tf.reshape(current,[B,F-1,H_,W_,2])
            
        warped_img = image_warp(other_imgs_dn,offsets)    
        warped_imglist = [image_warp(other_imgs_dn_suplist[i],offsets_list[i]) for i in range(len(offsets_list))]    
                   
    return warped_img,warped_imglist

def cost_volume(c1, warp, search_range, name,reuse):
    with tf.variable_scope(name,reuse=reuse): 
        padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
        _, h, w, _ = tf.unstack(tf.shape(c1))
        max_offset = search_range * 2 + 1
    
        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
                cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=3)
        cost_vol = lrelu(cost_vol)

    return cost_vol






def image_warp_3D_bil(images, flow, num_fr=5,name='image_warp'):
    ## Flow: B,H,W,3*(F-1)   images: B,H,W,C*(F-1)
    with tf.name_scope(name):
        shape = tf.shape(images)  
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        images = tf.transpose(tf.reshape(images,[batch_size,height,width,3,num_fr-1]),[0,1,2,4,3])  # B,H,W,F-1,C
        flow = tf.reshape(flow,[batch_size,height,width,3,num_fr-1])   # B,H,W,3,F-1

        offset_x = flow[:,:,:,0,:]   # B,H,W,F-1
        offset_y = flow[:,:,:,1,:]        
        offset_z = flow[:,:,:,2,:]
        
        x = tf.linspace(-1.0,1.0,width)
        y = tf.linspace(-1.0,1.0,height)
        
        x,y = tf.meshgrid(x,y)
        x,y = tf.reshape(x,[1,height,width,1]),tf.reshape(y,[1,height,width,1])        
        x_t, y_t = x + offset_x, y + offset_y   # B,H,W,F-1
        z_t = offset_z    # float
        
        max_x,max_y,max_z = width,height,num_fr-1 # int
        zero = tf.zeros([],tf.int32)    # int
        x_t_sc = 0.5*(x_t+1.0)*tf.cast(width-1,tf.float32)   #float
        y_t_sc = 0.5*(y_t+1.0)*tf.cast(height-1,tf.float32)        
        z_t_sc = 0.5*(z_t+1.0)*tf.cast(num_fr-2,tf.float32)   #float
  
        x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
        y1 = y0 + 1        
        z0 = tf.cast(tf.floor(z_t_sc),tf.int32)
        z1 = z0 + 1     
        
        x0,x1 = tf.clip_by_value(x0,zero,max_x),tf.clip_by_value(x1,zero,max_x)  #int
        y0,y1 = tf.clip_by_value(y0,zero,max_y),tf.clip_by_value(y1,zero,max_y)
        z0,z1 = tf.clip_by_value(z0,zero,max_z),tf.clip_by_value(z1,zero,max_z)   
        
        def get_pixel_value_3D(img,x,y,z):   # img B,H,W,F-1,C
            shape = tf.shape(x)  # x,y,z: B,H,W,F-1
            batch = shape[0]
            hei,wid,fr = shape[1],shape[2],shape[3] 
            batch_idx = tf.range(0,batch)
            batch_idx = tf.reshape(batch_idx,[batch,1,1,1])
            b = tf.tile(batch_idx,[1,hei,wid,fr])  # b: B,H,W,F-1
            indices = tf.stack([b,y,x,z],4)
            return tf.gather_nd(img,indices)     # B,H,W,F-1,C
        
        paddings = tf.constant([[0,0],[0,1],[0,1],[0,1],[0,0]])
        I000 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y0,z0)   # float
        I001 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y0,z1)
        I010 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y1,z0)
        I011 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y1,z1)   # B,H,W,C     
        I100 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y0,z0)   # B,H,W,C  
        I101 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y0,z1)   # B,H,W,C  
        I110 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y1,z0)   # B,H,W,C  
        I111 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y1,z1)   # B,H,W,C  
        
        
        x0,y0,z0 = tf.cast(x0,tf.float32),tf.cast(y0,tf.float32),tf.cast(z0,tf.float32)
        x1,y1,z1 = tf.cast(x1,tf.float32),tf.cast(y1,tf.float32),tf.cast(z1,tf.float32)

        w000 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z0-z_t_sc),0.0)
        w001 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z1-z_t_sc),0.0)
        w010 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z0-z_t_sc),0.0)
        w011 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z1-z_t_sc),0.0)
        w100 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z0-z_t_sc),0.0)
        w101 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z1-z_t_sc),0.0)
        w110 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z0-z_t_sc),0.0)
        w111 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)* \
                            tf.maximum(1.0-tf.abs(z1-z_t_sc),0.0)        
                    
        w000,w001,w010,w011 = tf.expand_dims(w000,4),tf.expand_dims(w001,4),tf.expand_dims(w010,4),tf.expand_dims(w011,4)
        w100,w101,w110,w111 = tf.expand_dims(w100,4),tf.expand_dims(w101,4),tf.expand_dims(w110,4),tf.expand_dims(w111,4)
        aligned_img = tf.add_n([w000*I000,w001*I001,w010*I010,w011*I011,w100*I100,w101*I101,w110*I110,w111*I111]) #B,H,W,F-1,C
        aligned_img = tf.transpose(aligned_img,[0,1,2,4,3])  #B,H,W,C,F-1
        aligned_img = tf.reshape(aligned_img,[batch_size,height,width,3*(num_fr-1)])

    return aligned_img     #B,H,W,C*(F-1)
def U_Net_align_spy_3D(img,num_fr=5,num_down=4,reuse = False):  # img:B,H,W,C,F  noise:B,H,W,1
    def pack_fea(x,num_fr=5): #B,F,H,W,C
        shape = tf.shape(x)
        B,H,W,C,F = shape[0],shape[1],shape[2],shape[3],shape[4]
        center = int((num_fr+1)/2)
        ref_img = x[:,:,:,:,center-1] #B,H,W,C
        
        inds = np.concatenate([np.arange(0,center-1),np.arange(center,num_fr)],0)
        others_list = [x[:,:,:,:,i] for i in inds]
        other_imgs = tf.stack(others_list,4)   ## B,H,W,C,F-1
        other_imgs = tf.reshape(other_imgs,[B,H,W,C*(F-1)])   ## B,H,W,(C*F-1)
        
        in_imgs = tf.concat([x[:,:,:,:,i] for i in range(num_fr)],3)
        
        return in_imgs,ref_img,other_imgs
        # in_imgs: B,H,W,C*F   ref_img: B,H,W,C   other_imgs: B,H,W,(C*F-1)
    def image_warp_3D(images, flow, name='image_warp'):
        ## Flow: B,H,W,3*(F-1)   images: B,H,W,C*(F-1)
        with tf.name_scope(name):
            shape = tf.shape(images)  
            batch_size = shape[0]
            height = shape[1]
            width = shape[2]
            images = tf.transpose(tf.reshape(images,[batch_size,height,width,3,num_fr-1]),[0,1,2,4,3])  # B,H,W,F-1,C
            flow = tf.reshape(flow,[batch_size,height,width,3,num_fr-1])   # B,H,W,3,F-1
    
            offset_x = flow[:,:,:,0,:]   # B,H,W,F-1
            offset_y = flow[:,:,:,1,:]        
            offset_z = flow[:,:,:,2,:]
            
            x = tf.linspace(-1.0,1.0,width)
            y = tf.linspace(-1.0,1.0,height)
            
            x,y = tf.meshgrid(x,y)
            x,y = tf.reshape(x,[1,height,width,1]),tf.reshape(y,[1,height,width,1])        
            x_t, y_t = x + offset_x, y + offset_y   # B,H,W,F-1
            z_t = offset_z    # float
            
            max_x,max_y,max_z = width,height,num_fr-2 # int
            zero = tf.zeros([],tf.int32)    # int
            x_t_sc = 0.5*(x_t+1.0)*tf.cast(width-1,tf.float32)   #float
            y_t_sc = 0.5*(y_t+1.0)*tf.cast(height-1,tf.float32)        
            z_t_sc = 0.5*(z_t+1.0)*tf.cast(num_fr-2,tf.float32)   #float
      
            x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
            y1 = y0 + 1        
            z0 = tf.cast(tf.round(z_t_sc),tf.int32)    
            
            x0,x1 = tf.clip_by_value(x0,zero,max_x),tf.clip_by_value(x1,zero,max_x)  #int
            y0,y1 = tf.clip_by_value(y0,zero,max_y),tf.clip_by_value(y1,zero,max_y)
            z0 = tf.clip_by_value(z0,zero,max_z)
            
            def get_pixel_value_3D(img,x,y,z):   # img B,H,W,F-1,C
                shape = tf.shape(x)  # x,y,z: B,H,W,F-1
                batch = shape[0]
                hei,wid,fr = shape[1],shape[2],shape[3] 
                batch_idx = tf.range(0,batch)
                batch_idx = tf.reshape(batch_idx,[batch,1,1,1])
                b = tf.tile(batch_idx,[1,hei,wid,fr])  # b: B,H,W,F-1
                indices = tf.stack([b,y,x,z],4)
                return tf.gather_nd(img,indices)     # B,H,W,F-1,C
            
            paddings = tf.constant([[0,0],[0,1],[0,1],[0,0],[0,0]])
            I00 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y0,z0)   # float
            I01 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x0,y1,z0)
            I10 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y0,z0)
            I11 = get_pixel_value_3D(tf.pad(images,paddings,"SYMMETRIC"),x1,y1,z0)   # B,H,W,C     
 
            
            
            x0,y0,z0 = tf.cast(x0,tf.float32),tf.cast(y0,tf.float32),tf.cast(z0,tf.float32)
            x1,y1 = tf.cast(x1,tf.float32),tf.cast(y1,tf.float32)

            w00 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0) 
            w01 = tf.maximum(1.0-tf.abs(x0-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)
            w10 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y0-y_t_sc),0.0) 
            w11 = tf.maximum(1.0-tf.abs(x1-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(y1-y_t_sc),0.0)

                        
            w00,w01,w10,w11 = tf.expand_dims(w00,4),tf.expand_dims(w01,4),tf.expand_dims(w10,4),tf.expand_dims(w11,4)
            aligned_img = tf.add_n([w00*I00,w01*I01,w10*I10,w11*I11]) #B,H,W,F-1,C
            aligned_img = tf.transpose(aligned_img,[0,1,2,4,3])  #B,H,W,C,F-1
            aligned_img = tf.reshape(aligned_img,[batch_size,height,width,3*(num_fr-1)])

        return aligned_img     #B,H,W,C*(F-1)
    def U_net2222(inputlist,other_list,ref_img_list,num_down=4,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,start_chan=32,act=lrelu,name=None,reuse=False):
        ## parameters
        # inputlist: B,H,W,C*F    other_list: B,H,W,C*(F-1)   ref_img_list: B,H,W,C
        chan_ = []

        if w_init is None:
            w_init = tf.contrib.slim.xavier_initializer()
        if b_init is None:
            b_init = tf.constant_initializer(value=0.0)    
        for i in range(num_down+1):
            chan_.append(start_chan*(2**(i)))
            ##
        warping_func = image_warp_3D
        multis_list = []
        temporal_list = []
        with tf.variable_scope('local_ops',reuse=reuse):
            current = inputlist[num_down]
            for j in range(num_conv):
                current = slim.conv2d(current,chan_[num_down],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, rate=1,activation_fn=act,padding='SAME',scope='g_conv_block_%d'%(j),reuse=reuse)
            fine_flow = slim.conv2d(current,3*(num_fr-1),[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init,activation_fn=tf.nn.tanh,padding='SAME',scope='g_conv_block',reuse=reuse)
            # B,H,W,F-1
            temporal_list.append(tf.reshape(fine_flow,[fine_flow.shape[0],fine_flow.shape[1],fine_flow.shape[2],3,num_fr-1])) 
            
            
            other_imgs = other_list[num_down]
            '''
            shape_cur = tf.shape(fine_flow)
            B_c,H_c,W_c = shape_cur[0],shape_cur[1],shape_cur[2]
            fine_flow0 = tf.zeros_like(fine_flow[:,:,:,0:num_fr-1],dtype=tf.float32)
            fine_flow1 = tf.zeros_like(fine_flow[:,:,:,0:num_fr-1],dtype=tf.float32)
            fine_flow2 = tf.ones_like(fine_flow[:,:,:,0:num_fr-1],dtype=tf.float32)*tf.reshape(tf.constant([-1.0,-0.8,0.8,1.0]),[1,1,1,4])
            fine_flow = tf.stack([fine_flow0,fine_flow1,fine_flow2],3)
            fine_flow = tf.reshape(fine_flow,[B_c,H_c,W_c,-1])
            '''
            multis_list.append(warping_func(other_imgs,fine_flow,name='fine_wapring%d'%(num_down)))  
            restore_temp = fine_flow

        
        with tf.variable_scope('expanding_ops',reuse=reuse):
            init_flow = restore_temp
            for i in range(num_down):
                index_current = num_down-1-i
                other_imgs,ref_img = other_list[index_current],ref_img_list[index_current]                    
                up_flow = slim.conv2d_transpose(init_flow,3*(num_fr-1),3,(2,2),padding='SAME',scope='up%d'%(i),reuse=reuse)

                other_img_warped = image_warp_3D_bil(other_imgs,up_flow,name='init_wapring%d'%(i))
                fea = tf.concat([other_img_warped,ref_img,up_flow],3)
            
                #current =  upsample_and_concat_c( current, conv_[index_current], chan_[index_current], chan_[index_current+1], scope='uac%d'%(i),reuse=reuse )
                current = slim.conv2d(fea,chan_[index_current],[fil_s,fil_s],
                                      weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=act,scope='g_dconv%d'%(i),reuse=reuse)
                for j in range(num_conv):
                    current=slim.conv2d(current, chan_[index_current],[fil_s,fil_s], 
                                        weights_initializer=w_init,biases_initializer=b_init,rate=1, padding='SAME',activation_fn=act,scope='g_dconv_block%d_%d'%(i,j),reuse=reuse)
                fine_flow = up_flow + slim.conv2d(current,3*(num_fr-1),[fil_s,fil_s], 
                                    weights_initializer=w_init,biases_initializer=b_init, padding='SAME',activation_fn=tf.nn.tanh,scope='g_dconv_block%d'%(i),reuse=reuse)
                
                temporal_list.append(tf.reshape(fine_flow,[fine_flow.shape[0],fine_flow.shape[1],fine_flow.shape[2],3,num_fr-1])) 
                multis_list.append(warping_func(other_imgs,fine_flow,name='fine_wapring%d'%(i)))    
                init_flow = fine_flow
             
    
        return multis_list,temporal_list  
    
        
    shape = tf.shape(img)
    B,H,W,C,F = shape[0],shape[1],shape[2],shape[3],shape[4]

    with tf.variable_scope('Stage1',reuse=reuse):
        current = img                
        in_imgs,ref_img,other_imgs = pack_fea(current,num_fr=num_fr)  #B*(F-1),H,W,C*2, #B,1,H,W,C #B,F-1,H,W,C
        in_imgs = tf.clip_by_value(in_imgs,0.0,1.0)
        inlist = []
        ref_img_list = []
        other_list = []
        for i in range(num_down+1):
            size = [tf.cast(H/(2**i),tf.int32),tf.cast(W/(2**i),tf.int32)]            
            inlist.append(tf.image.resize_bilinear(in_imgs,size))
            other_list.append(tf.image.resize_bilinear(other_imgs,size))   
            ref_img_list.append(tf.image.resize_bilinear(ref_img,size))   

        warped_imgs_list,temporal_list = U_net2222(inlist,other_list,ref_img_list,num_down=num_down,num_conv=6,num_out=2,fil_s=5,w_init=None,b_init=None,
                                start_chan=32,act=lrelu,name='Alignment',reuse=reuse)  # B*(F-1),H,W,2 
        
        for i in range(len(warped_imgs_list)):
            current = warped_imgs_list[i]
            shape = tf.shape(current)
            H_c,W_c,C_c = shape[1],shape[2],shape[3]
            warped_imgs_list[i] = tf.reshape(current,[B,H_c,W_c,3,F-1])
        warped_imgs_list.reverse()   
        temporal_list.reverse()
    return warped_imgs_list,temporal_list









'''
img = tf.zeros(shape=[1,512,512,3,7])
out = EDVR(img,fra_s=7,num_fea=32,reuse = False)
'''
'''
img = tf.zeros(shape=[1,512,512,3,7])
noise = tf.zeros(shape=[1,512,512,1])
out = LiangNN(img,noise,num_fr=7,reuse = False)

img = tf.ones(shape=[1,512,512,3*7])
w_init = tf.constant_initializer(0.0)
b_init = tf.constant_initializer(0.0)
offsets = U_net22(img,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=w_init,b_init=b_init,is_residual=False,
                                start_chan=32,act=lrelu,is_global=True,name='Alignment',reuse=False)  # B,H,W,3
out = tf.reduce_mean(offsets)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
out_ = sess.run(out)
print(out_)
'''
def STTN(img,noise,num_fr=5,reuse = False):  # img:B,H,W,C,F  noise:B,H,W,1
    shape = tf.shape(img)
    B,H,W,C,F = shape[0],shape[1],shape[2],shape[3],shape[4]
    img_in = tf.reshape(img,[B,H,W,C*F])  #B,H,W,C*F
    
    ### Alignment
    with tf.variable_scope('STTN',reuse=reuse):
        current = img_in
        current = (tf.clip_by_value(current,0.0,1.0)+1e-4)**(1.0/2.2)
        w_init = None#tf.constant_initializer(0.0)
        b_init = None#tf.constant_initializer(0.0)
        offsets = U_net22(current,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=w_init,b_init=b_init,is_residual=False,
                                start_chan=32,act=lrelu,is_global=True,name='Alignment',reuse=reuse)  # B,H,W,3
        img_to_align = tf.transpose(img,[0,1,2,4,3])  # img:B,H,W,F,C
        
        offset_x = offsets[:,:,:,0]   # B,H,W
        offset_y = offsets[:,:,:,1]        
        offset_z = offsets[:,:,:,2]
        
        x = tf.linspace(-1.0,1.0,W)
        y = tf.linspace(-1.0,1.0,H)
        
        x,y = tf.meshgrid(x,y)
        x,y = tf.reshape(x,[-1,H,W]),tf.reshape(y,[-1,H,W])        
        x_t, y_t = x + offset_x, y + offset_y
        z_t = offset_z
        
        max_x,max_y,max_z = W-1,H-1,F-1 # int
        zero = tf.zeros([],tf.int32)    # int
        x_t_sc = 0.5*(x_t+1.0)*tf.cast(max_x,tf.float32)   #float
        y_t_sc = 0.5*(y_t+1.0)*tf.cast(max_y,tf.float32)        
        z_t_sc = 0.5*(z_t+1.0)*tf.cast(max_z,tf.float32)   #float
  
        x0 = tf.cast(tf.floor(x_t_sc),tf.int32)   # int
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y_t_sc),tf.int32)
        y1 = y0 + 1        
        z0 = tf.cast(tf.floor(z_t_sc),tf.int32)
        z1 = z0 + 1     
        
        x0,x1 = tf.clip_by_value(x0,zero,max_x),tf.clip_by_value(x1,zero,max_x)  #int
        y0,y1 = tf.clip_by_value(y0,zero,max_y),tf.clip_by_value(y1,zero,max_y)
        z0,z1 = tf.clip_by_value(z0,zero,max_z),tf.clip_by_value(z1,zero,max_z)        
        
        I000 = get_pixel_value_3D(img_to_align,x0,y0,z0)   # float
        I001 = get_pixel_value_3D(img_to_align,x0,y0,z1)
        I010 = get_pixel_value_3D(img_to_align,x0,y1,z0)
        I011 = get_pixel_value_3D(img_to_align,x0,y1,z1)   # B,H,W,C     
        I100 = get_pixel_value_3D(img_to_align,x1,y0,z0)   # B,H,W,C  
        I101 = get_pixel_value_3D(img_to_align,x1,y0,z1)   # B,H,W,C  
        I110 = get_pixel_value_3D(img_to_align,x1,y1,z0)   # B,H,W,C  
        I111 = get_pixel_value_3D(img_to_align,x1,y1,z1)   # B,H,W,C  
        
        w000 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w001 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w010 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w011 = tf.maximum(1.0-tf.abs(tf.cast(x0,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w100 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w101 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y0,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)
        w110 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z0,tf.float32)-z_t_sc),0.0)
        w111 = tf.maximum(1.0-tf.abs(tf.cast(x1,tf.float32)-x_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(y1,tf.float32)-y_t_sc),0.0)*tf.maximum(1.0-tf.abs(tf.cast(z1,tf.float32)-z_t_sc),0.0)        
        
        w000,w001,w010,w011 = tf.expand_dims(w000,3),tf.expand_dims(w001,3),tf.expand_dims(w010,3),tf.expand_dims(w011,3)
        w100,w101,w110,w111 = tf.expand_dims(w100,3),tf.expand_dims(w101,3),tf.expand_dims(w110,3),tf.expand_dims(w111,3)
        aligned_img = tf.add_n([w000*I000,w001*I001,w010*I010,w011*I011,w100*I100,w101*I101,w110*I110,w111*I111])
        
    ### Fusion
    with tf.variable_scope('ImageProcessing',reuse=reuse):
        current = tf.concat([aligned_img,noise],3)
        out_final = U_net22(current,num_down=4,num_block=1,num_conv=1,num_out=3,rate=[1]*10,fil_s=3,w_init=None,b_init=None,is_residual=False,
                                start_chan=32,act=lrelu,is_global=True,name='IP',reuse=reuse)  # B,H,W,3    
    return aligned_img,out_final




    
    
    
    
    
    