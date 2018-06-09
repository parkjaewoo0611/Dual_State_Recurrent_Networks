import numpy as np
import os
import pickle
import argparse
from PIL import Image

# parsing cmd arguments
parser = argparse.ArgumentParser(description="Run commands")
def str2bool(v):
    return v.lower() == 'true'
parser.add_argument('--input_path', default = '../../data/image/val', type = str,
                    help = "train data image path")

parser.add_argument('--output_file', default = '../../data/patch/val/val_HR.npy', type = str,
                    help = 'train data patch path')

parser.add_argument('--patch_size', default = 128, type = int,
                    help = "patch size of HR")

parser.add_argument('--augmentation', default = True, type = bool,
                    help = "data augmentation")

def im2patch(im, shape, stride):
    H, W, C = im.shape
    HR_patchs = np.ndarray([1, shape[0], shape[1], C])

    i = 0
    j = 0
    while stride[0] * (i + 1) < H:
        while stride[1] * (j + 1)< W:
            HR_patch = im[stride[0] * i: stride[0] * (i + 1), stride[1] * j: stride[1] * (j + 1), :]
            HR_patch = np.reshape(HR_patch, [1, shape[0], shape[1], C])
            HR_patchs = np.append(HR_patchs, HR_patch, 0)
            j = j + 1
        i = i + 1
    HR_result = HR_patchs[1:, :, :, :].astype(np.float32)
    return HR_result

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    input_path = args.input_path

    filelist = os.listdir(input_path)
    hr_size = args.patch_size

    HR = np.zeros([1, hr_size, hr_size])

    for f in filelist:
       HR_rgb = Image.open(os.path.join(args.input_path, f))
       HR_ycbcr = rgb2ycbcr(np.array(HR_rgb))
       HR_patch = im2patch(HR_ycbcr, [hr_size, hr_size], [hr_size, hr_size])
       HR_patch = HR_patch[:,:,:,0]

       HR = np.concatenate([HR, HR_patch], axis = 0)

       if args.augmentation == True:
           augment1 = [Image.FLIP_LEFT_RIGHT,
                       Image.FLIP_TOP_BOTTOM,
                       Image.ROTATE_90,
                       Image.ROTATE_180,
                       Image.ROTATE_270]
           for method in augment1:
               HR_rotated_rgb = HR_rgb.transpose(method)
               HR_rotated_ycbcr = rgb2ycbcr(np.array(HR_rotated_rgb))
               HR_rotated_patch = im2patch(HR_rotated_ycbcr, [hr_size, hr_size], [hr_size, hr_size])
               HR_rotated_patch = HR_rotated_patch[:,:,:,0]

               HR = np.concatenate([HR, HR_rotated_patch], axis = 0)

           augment2 = [0.5, 0.6, 0.7, 0.8, 0.9]
           for i in augment2:
               HR_scaled_rgb = HR_rgb.resize([int(hr_size * i), int(hr_size * i)], Image.BICUBIC)
               HR_scaled_ycbcr = rgb2ycbcr(np.array(HR_scaled_rgb))
               HR_scaled_patch = im2patch(HR_scaled_ycbcr, [hr_size, hr_size], [hr_size, hr_size])
               HR_scaled_patch = HR_scaled_patch[:,:,:,0]

               HR = np.concatenate([HR, HR_scaled_patch], axis = 0)
    np.save(args.output_file, HR[1:,:,:])


