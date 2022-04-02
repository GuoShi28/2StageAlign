import numpy as np
import glob
import math
import random
import os
import cv2
from PIL import Image

def rgf_rgb_img_save(img, path, name, x, y):
    img_wr = np.clip(img * 255.0, 0.0, 255.0)
    img_wr = img_wr.astype(np.uint8)
    for i in range(5):
        patch_name = path + '/' + name + '_' + str(x) + '_' + str(y) + '_' + str(i) + '.png'
        rgf1 = img_wr[:, :, 3 * i: 3 * (i + 1)]
        rgf2 = cv2.cvtColor(rgf1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_name, rgf2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def find_large_motion_jpg(rgb_patch, x, y, jpg_tmp_save_path):
    h = rgb_patch.shape[0]
    w = rgb_patch.shape[1]
    l = rgb_patch.shape[2]//3
    y_patch = np.zeros((h, w, l))
    y_patch = y_patch.astype(np.uint8)
    for i in range(l):
        y_arr = 0.213 * rgb_patch[:, :, i*3] + 0.715 * rgb_patch[:, :, i*3+1] + 0.072 * rgb_patch[:, :, i*3+2]
        y_arr = y_arr.astype(np.uint8)
        y_patch[:, :, i] = y_arr
        #y_patch_name = jpg_tmp_save_path + '/' + 'y' + '_' + str(x) + '_' + str(y) + '_' + str(i) + '.png'
        #cv2.imwrite(y_patch_name, y_patch[:, :, i], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if i > 0:
            diff = np.abs(y_patch[:, :, i] - y_patch[:, :, i-1])
            diff_avg = np.mean(diff)
            # print('diff avg is %d' %diff_avg)
            if diff_avg > 130:
                print('The motion between two frames is too big to discard!!!')
                return False

    return True

def find_flat_patch(rgb_patch, flat_thd = 5):
    l = rgb_patch.shape[2] // 3
    for i in range(l):
        r_arr = rgb_patch[:, :, i * 3]
        g_arr = rgb_patch[:, :, i * 3 + 1]
        b_arr = rgb_patch[:, :, i * 3 + 2]
        if ((np.std(r_arr) < flat_thd) and (np.std(g_arr) < flat_thd) and (np.std(b_arr) < flat_thd)):
            print('The patch is too flat to abandon!!!')
            return False

    return True

def random_small_motion_rgb_patch(rgb_single, patch_size, small_motion_max, x, y, nr_frames_num_small_motion):
    x_c = (x + x + patch_size + small_motion_max*2)//2
    y_c = (y + y + patch_size + small_motion_max*2) // 2
    x_new = x_c + random.randint(-small_motion_max, small_motion_max)
    y_new = y_c + random.randint(-small_motion_max, small_motion_max)
    rgb_new = rgb_single[int(x_new-(patch_size//2)): int(x_new+(patch_size//2)), int(y_new-(patch_size//2)): int(y_new+(patch_size//2)), :]
    for i in range(1, nr_frames_num_small_motion):
        x_new = x_c + random.uniform(-small_motion_max, small_motion_max)
        y_new = y_c + random.uniform(-small_motion_max, small_motion_max)
        rgb_new_img = rgb_single[int(x_new-(patch_size//2)): int(x_new+(patch_size//2)), int(y_new-(patch_size//2)): int(y_new+(patch_size//2)), :]
        rgb_new = np.concatenate([rgb_new, rgb_new_img], axis=2)

    return rgb_new

def jpg_to_rawGT(jpg_tmp_save_path, jpg_train_out_path, nr_frames_num, patch_size, count):
    jpg_list = glob.glob(jpg_tmp_save_path + "/*.png")
    jpg_list = sorted(jpg_list)
    if len(jpg_list) != nr_frames_num:
        return False

    judge_smallmove = random.uniform(0, 1) > 0.5
    img_multi = cv2.imread(jpg_list[0], cv2.IMREAD_UNCHANGED)
    #cv2.imshow('rgf0', img_multi)
    rgb_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB)
    rgb_single = rgb_multi
    #cv2.imshow('rgf1', img_multi)
    height = rgb_multi.shape[0]
    width = rgb_multi.shape[1]
    for i in range(1, len(jpg_list)):
        img = cv2.imread(jpg_list[i], cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_multi = np.concatenate([rgb_multi,rgb],axis=2)
        # rgb2 = cv2.resize(rgb, (height//2, width//2))

    # rgb_multi.shape = (1080, 1920, 3*nr_frames_num)
    small_motion_max = 15
    # stride = patch_size + small_motion_max*2
    pos_list = []
    for x in range(6, height - 6 - patch_size, patch_size):
        for y in range(6, width - 6 - patch_size, patch_size):
            pos = [x, y]
            pos_list.append(pos)

    
    for x, y in pos_list:
        if judge_smallmove:
            print("Current coordinate is [%d, %d] with smallmove" %(x, y))
            rgb_patch = random_small_motion_rgb_patch(rgb_single, patch_size, small_motion_max, x, y, nr_frames_num)
        else:
            print("Current coordinate is [%d, %d] with dynamic" %(x, y))
            rgb_patch = rgb_multi[x: x+patch_size, y: y+patch_size, :]
        
        
        is_not_large = find_large_motion_jpg(rgb_patch, x, y, jpg_tmp_save_path)
        if (is_not_large):
            is_not_flat = find_flat_patch(rgb_patch,flat_thd= 3)
            if (is_not_flat):
                count = count + 1
                save_patch_path = jpg_train_out_path + '/' + str(count)
                if not os.path.exists(save_patch_path):
                    os.makedirs(save_patch_path)

                for i in range(nr_frames_num):
                    #cv2.imshow('rgf2', rgb_patch[:,:,3*i : 3*(i+1)])
                    #cv2.waitKey()
                    rgb_patch_name = save_patch_path + '/' + str(i).zfill(2) + '.png'
                    rgf1 = rgb_patch[:,:,3*i : 3*(i+1)]
                    rgf2 = cv2.cvtColor(rgf1, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(rgb_patch_name, rgf2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return count

def rgf_video_to_rawGT(video_list, jpg_tmp_out_path, jpg_train_out_path, nr_frames_num, patch_size, skip_frames_num):
    count = 0
    for i in range(len(video_list)):
        vcap = cv2.VideoCapture(video_list[i])
        frame_nums = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("%d video with %d frames is under processing!" %(i, frame_nums))
        suc = vcap.isOpened()
        if not suc:
            print("This video is invalid!")
            break

        video_tmp_save_path = jpg_tmp_out_path + "/" + str(i)
        if not os.path.exists(video_tmp_save_path):
            os.makedirs(video_tmp_save_path)

        skip_num = int(nr_frames_num + skip_frames_num)
        skip_idx = 0
        for j in range(frame_nums-skip_num):
            suc, frame = vcap.read()
            if not suc:
                break
            if (j < skip_num):
                continue
            if j%skip_num < nr_frames_num:
                if j%skip_num == 0:
                    jpg_tmp_save_path = video_tmp_save_path + "/" + str(j)
                    if not os.path.exists(jpg_tmp_save_path):
                        os.makedirs(jpg_tmp_save_path)
                cv2.imwrite(jpg_tmp_save_path + "/" + str(j) + '.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            if j%skip_num == nr_frames_num:
                print('%d frame will be converted to raw' %j)
                count = jpg_to_rawGT(jpg_tmp_save_path, jpg_train_out_path, nr_frames_num, patch_size, count)


if __name__ == "__main__":
    video_in_path = "/home/guoshi/data6T/Vivo_Damo/Original_videos/"
    jpg_tmp_out_path = "/home/guoshi/data6T/Vivo_Damo/NewTrainingData/temp/"
    jpg_train_out_path = "/home/guoshi/data6T/Vivo_Damo/NewTrainingData/TrainPatches/"
    nr_frames_num = 9
    patch_size = 256
    skip_frames_num = 200

    video_list  = glob.glob(video_in_path + "/*.mkv")
    print('Totally %d videos are under processing!' %(len(video_list)))

    if not os.path.exists(jpg_tmp_out_path):
        os.makedirs(jpg_tmp_out_path)

    rgf_video_to_rawGT(video_list, jpg_tmp_out_path, jpg_train_out_path, nr_frames_num, patch_size, skip_frames_num)



