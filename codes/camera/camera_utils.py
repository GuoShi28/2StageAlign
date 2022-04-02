import math
import torch
import cv2
import torch.nn as nn
import numpy as np

'''
Implement for Camera base shift.
This Code is refer to released GeoNet code.

Author: Guo Shi
Time: 2019.09.18 
'''

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to camera frame
    Args:
        depth: [batch, height, width]
        pixel_coords: homogeneous pixel coordinates [batch, 3, height, weight]
        instrinsics: camera intrinsics [batch, 3, 3]
        is_homogeneous: return in homogeneous coordinates
    Returns:
        Coords in the camera frame [batch, 3 (4 is homogeneous), height, width]
    """
    batch, height, width = depth.shape
    depth = torch.reshape(depth, (batch, 1, -1)) # -> B, 1, H*W
    pixel_coords = torch.released(pixel_coords, (batch, 3, -1)) # -> B, 3, H*W
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth # B, 3, H*W
    if is_homogeneous:
        ones = torch.ones(batch, 1, height, width)
        if depth.is_cuda:
            ones = ones.cuda()
        cam_coords = torch.cat((cam_coords, ones), 1)
    cam_coords = torch.reshape(cam_coords, (batch, -1, height, width)) # [batch, 3 (4 is homogeneous), height, width]
    return cam_coords

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame
    Args:
        cam_coords: [batch, 4, height, width]
        proj: [batch, 4, 4]
    Return:
        Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, [batch, 4, -1]) # B, 4, H*W
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords) # B, 4, H*W
    x_u = unnormalized_pixel_coords[:, 0:1, :] # B,1,H*W
    y_u = unnormalized_pixel_coords[:, 1:2, :] # B,1,H*W
    z_u = unnormalized_pixel_coords[:, 2:3, :] # B,1,H*W
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = torch.cat((x_n, y_n), 1) # B,2,H*W
    pixel_coords = torch.transpose(pixel_coords, 1, 2) # B, H*W, 2
    pixel_coords = torch.reshape(pixel_coords, (batch, height, width, 2)) # B,H,W,2
    # why trahsfer to B*W*H*2, Does this for TF training???
    return pixel_coords

def meshgrid(batch, height, width, is_homogeneous=True):
    """Constract a 2D meshgrid
    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2(3 if homogeneous), height, width]
    """
    temp = torch.ones(height, 1) # H,1
    temp2 = torch.linspace(-1, 1, step=width)
    temp2 = torch.reshape(temp2, (width, 1)) # W,1
    temp2 = torch.transpose(temp2, 0, 1) # 1,W
    x_t = torch.matmul(temp, temp2) # H, W
    x_t = torch.reshape(x_t, (1, height, width)) # 1, H, W

    temp = torch.linspace(-1, 1, step=height)
    temp = torch.reshape(temp, (height, 1)) # H, 1
    temp2 = torch.ones(1, width) # 1, W
    y_t = torch.matmul(temp, temp2)
    y_t = torch.reshape(y_t, (1, height, width)) # 1, H, W

    x_t = (x_t + 1.0) * 0.5 * torch.float(width-1)
    y_t = (y_t + 1.0) * 0.5 * torch.float(height-1)
    if is_homogeneous:
        ones = torch.ones_like(x_t)
        coords = torch.cat((x_t, y_t, ones), 0) # 3, H, W
    else:
        coords = torch.cat((x_t, y_t), 0)
        
    coords = torch.unsqueeze(coords, 0) # 1, 2(3 if is_homogeneous), H, W
    coords = coords.repeat(batch, 1, 1, 1) # B, 2(3 if is_homogeneous), H, W
    return coords

def flow_warp(src_img, flow):
    """ inverse wrap a source image to the target image plane based on flow field
    Args:
        src_img: source image [batch, 3, height_s, width_s]
        flow: target image to source image flow [batch, 2, height_t, width_t]
    Return:
        Source image inverse wrapped to the target image plane [batch, 3, height_t, width_t]
    """
    batch, _, height, width = src_img.shape
    tgt_pixel_coords = meshgrid(batch, height, width, False) # B, 2, H, W
    src_pixel_coords = tgt_pixel_coords + flow
    output_img = bilinear_sampler(src_img, src_pixel_coords)
    return output_img

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
    """Compute the rigid flow from the target image plane to source image
    Args:
        depth: depth map of the target image [batch, height_t, width_t]
        pose: target to source (or source to target if reverse_pose=True)
              camera transformation matrix [batch, 6], in the order of
              tx, ty, tz, rx, ry, rz
        intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
        Rigid flow from target image to source image [batch, height_t, width_t, 2]
    """
    batch, height, width = depth.shape
    # convert pose vector to matrix
    pose = pose_vec2mat(pose) # Batch, 4, 4
    if reverse_pose:
        pose = torch.inverse(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width) # B, 3, H, W
    tgt_pixel_coords = pixel_coords[:,:2,:,:] # B, 2, H, W

    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4*4 intrinsic matrix
    filler = torch.tensor([0.0, 0.0, 0.0, 1.0])
    filler = torch.reshape(filler, (1, 1, 4))
    filler = filler.repeat(batch, 1, 1) # B, 1, 4
    intrinsics = torch.cat((intrinsics, torch.zeros(batch, 3, 1)), 2) # B, 3, 4
    intrinsics = torch.cat((intrinsics, filler), 1) # B, 4, 4

    # Get a 4*4 transformation matrix from 'target' camera frame to 'source' pixel frame
    # pixel frame
    proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, pose) # B, 4, 4
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    rigid_flow = src_pixel_coords - tgt_pixel_coords
    return rigid_flow

def bilinear_sampler(img, coords):
    """Construct a new image by bilinear sampling from the input image
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, channels, height_s, width_s]
        coords: coordinates of source pixels to sample from [batch, 2, height_t, width_t]
        . height_t/width_t correspond to the dimensions of the output image
        (don't need to be the same as height_s/width_s). 
        The two channels correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, channels, height_t, width_t]
    """
    def _repeat(x, n_repeat):
        temp = torch.ones(n_repeat)
        temp = temp.unsqueeze(1)
        temp = torch.transpose(temp, 1, 0)
        rep = torch.float(temp)
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.reshape(x, [-1])
    

    # bilinear process
    coords_x = coords[:,0,:,:] # B, out_H, out_W
    coords_y = coords[:,1,:,:] # B, out_H, out_W
    batch, inp_ch, inp_h, inp_w = img.shape # B, C, in_H, in_W
    _, _, out_h, out_w = coords_x.shape
    coords_x = torch.float(coords_x)
    coords_y = torch.float(coords_y)

    x0 = torch.floor(coords_x) # largest integer less than coords_x
    x1 = x0 + 1
    y0 = torch.floor(coords_y)
    y1 = y0 + 1

    y_max = torch.float(inp_h - 1)
    x_max = torch.float(inp_w - 1)
    zero = torch.float(1.0)

    x0_safe = torch.clamp(x0, zero, x_max)
    y0_safe = torch.clamp(y0, zero, y_max)
    x1_safe = torch.clamp(x1, zero, x_max)
    y1_safe = torch.clamp(y1, zero, y_max)

    # bilinear interp weight, with points outside the grid weight 0
    wt_x0 = x1_safe - coords_x # B, out_H, out_W
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    # indices in the flat image to sample from
    dim2 = torch.float(inp_w)
    dim1 = torch.float(inp_h*inp_w)
    temp = torch.range(batch) * dim1 # 0~batch-1 * (W*H)
    temp = _repeat(temp, out_h*out_w)
    base = torch.reshape(temp, (batch, 1, out_h, out_w))

    base_y0 = base + y0_safe * dim2 
    base_y1 = base + y1_safe * dim2
    idx00 = torch.reshape(x0_safe + base_y0, (-1)) # B*out_H*out_W
    idx01 = torch.reshape(x0_safe + base_y1, (-1)) # B*out_H*out_W
    idx10 = torch.reshape(x1_safe + base_y0, (-1)) # B*out_H*out_W
    idx11 = torch.reshape(x1_safe + base_y1, (-1)) # B*out_H*out_W

    ## sample from images
    img_temp = torch.reshape(img, (batch, inp_ch, -1))
    img_temp = torch.transpose(img_temp, 2, 1) # B, H*W, C
    imgs_flat = torch.reshape(imgs, (-1, inp_ch)) # B*H*W, C
    imgs_flat = torch.float(imgs_flat)
    im00_temp = torch.index_select(imgs_flat, 0, torch.int(idx00)) # B*out_H*out_W, C
    im00 = torch.reshape(im00_temp, (batch, out_h, out_w, inp_ch)) # B, out_H, out_W, C
    im01_temp = torch.index_select(imgs_flat, 0, torch.int(idx01)) # B*out_H*out_W, C
    im01 = torch.reshape(im01_temp, (batch, out_h, out_w, inp_ch)) # B, out_H, out_W, C
    im10_temp = torch.index_select(imgs_flat, 0, torch.int(idx10)) # B*out_H*out_W, C
    im10 = torch.reshape(im10_temp, (batch, out_h, out_w, inp_ch)) # B, out_H, out_W, C
    im11_temp = torch.index_select(imgs_flat, 0, torch.int(idx11)) # B*out_H*out_W, C
    im11 = torch.reshape(im11_temp, (batch, out_h, out_w, inp_ch)) # B, out_H, out_W, C

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11 # Does exit the broadcast problem???
    # Assume to be B, out_H, out_W, C 
    output = torch.reshape(output, (batch, -1, inp_ch))
    output = torch.transpose(output, 2, 1)
    output = torch.reshape(output, (batch, inp_ch, out_h, out_w))
    return output

def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [b, 6]
    Return:
        Atransformation matrix -- [B, 4, 4]
    """
    batch, _ = vec.shape
    translation = vec[:, 0:3] # B, 3
    rx = vec[:, 3:4] # B, 1
    ry = vec[:, 4:5] # B, 1
    rz = vec[:, 5:6] # B, 1
    rot_mat = euler2mat(rz, ry, rx) # B, 1, 3, 3
    rot_mat = rot_mat.squeeze(1) # B, 3, 3
    filler = torch.tensor([0.0, 0.0, 0.0, 1.0])
    filler = torch.reshape(filler, (1, 1, 4)) # 1, 1, 4
    transform_mat = torch.cat((rot_mat, translation.unsqueeze(2)), 2) # B, 3, 4
    transform_mat = torch.cat((transform_mat, filler), 1) # B, 4, 4 

    return rot_mat

def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
    TODO: remove the dimension for 'N' (deprecated for converting all source
            pose altogether)
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    batch = z.shape[0]
    N = 1
    z = torch.clamp(z, -np.pi, np.pi)
    y = torch.clamp(y, -np.pi, np.pi)
    x = torch.clamp(x, -np.pi, np.pi)

    # Expand to B, N, 1, 1
    z = z.unsqueeze(2)
    z = z.unsqueeze(3)
    y = y.unsqueeze(2)
    y = y.unsqueeze(3)
    x = x.unsqueeze(2)
    x = x.unsqueeze(3)

    zeros = torch.zeros(batch, N, 1, 1)
    ones = torch.ones(batch, N, 1, 1)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    rotz_1 = torch.cat((cosz, -sinz, zeros), 3) # B, N, 1, 3
    rotz_2 = torch.cat((sinz,  cosz, zeros), 3) # B, N, 1, 3
    rotz_3 = torch.cat((zeros, zeros, ones), 3) # B, N, 1, 3
    zmat = torch.cat((rotz_1, rotz_2, rotz_3), 2) # B, N, 3, 3

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_1 = torch.cat((cosy, zeros, siny), 3) # B, N, 1, 3
    roty_2 = torch.cat((zeros, ones, zeros), 3) # B, N, 1, 3
    roty_3 = torch.cat((-siny, zeros, cosy), 3) # B, N, 1, 3
    ymat = torch.cat((roty_1, roty_2, roty_3), 2) # B, N, 3, 3

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_1 = torch.cat((ones, zeros, zeros), 3) # B, N, 1, 3
    rotx_2 = torch.cat((zeros, cosx, -sinx), 3) # B, N, 1, 3
    rotx_3 = torch.cat((zeros, sinx, cosx), 3) # B, N, 1, 3
    xmat = torch.cat((rotx_1, rotx_2, rotx_3), 2) # B, N, 3, 3

    rotMat_temp = torch.matmul(xmat, ymat) # B, N, 3, 3
    rotMat = torch.matmul(rotMat_temp, zmat)
    return rotMat 



    













 







