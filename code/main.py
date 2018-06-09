import argparse
import sys
import os
import glob
from data_loader import *
from SR import *
from patch_generator import *
from scipy.misc import imsave
# parsing cmd arguments
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--is_train', default=True			     , type=str2bool)
parser.add_argument('--model'   , default='DRRN'		     , type=str)
parser.add_argument('--gpu'     , default="0"			     , type=str)
parser.add_argument('--scale'   , default=0.5			     , type=float)

def main():
    args, unparsed = parser.parse_known_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    load_dir = 'logs/' + args.model + '_' + str(int(1/args.scale))
    if args.is_train:
        HR, LR, bicubic_LR = data_loader('../data/patch/train/train_HR.npy', args.scale)
        val_HR, val_LR, val_bicubic_LR = data_loader('../data/patch/val/val_HR.npy', args.scale)

        if args.model == 'DSRN':
            with tf.Session() as sess:
                SR = SuperResolution(args)
                SR.train(sess, args, LR, HR, val_LR, val_HR)
        else:
            with tf.Session() as sess:
                SR = SuperResolution(args)
                SR.train(sess, args, bicubic_LR, HR, val_bicubic_LR, val_HR)
    else:
        file_list1 = glob.glob('../data/image/test/BSD100/*.*')
        file_list2 = glob.glob('../data/image/test/Set5/*.*')
        file_list3 = glob.glob('../data/image/test/Set14/*.*')
        file_list4 = glob.glob('../data/image/test/Urban100/*.*')
        result_path1 = '../data/image/result/'+args.model+'_'+str(int(1/args.scale))+'/BSD100'
        result_path2 = '../data/image/result/'+args.model+'_'+str(int(1/args.scale))+'/Set5'
        result_path3 = '../data/image/result/'+args.model+'_'+str(int(1/args.scale))+'/Set14'
        result_path4 = '../data/image/result/'+args.model+'_'+str(int(1/args.scale))+'/Urban100'
        tf.reset_default_graph()

        with tf.Session() as sess:
            SR = SuperResolution(args)
            SR.saver.restore(sess, load_dir + '/model.ckpt')
            for file in file_list1:
                HR_rgb = Image.open(file)
                LR_rgb = np.clip(imresize(HR_rgb, args.scale, interp='bicubic'),0,255.0)

                LR_ycbcr = rgb2ycbcr(LR_rgb)
                LR_y = LR_ycbcr[:,:,0]
                bicubic_LR_cb = np.clip(imresize(LR_ycbcr[:,:,1], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_cr = np.clip(imresize(LR_ycbcr[:,:,2], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_y = np.clip(imresize(LR_y, 1/args.scale, interp='bicubic'),0,255.0)
                if args.model is not 'DSRN':
                    SR_y = SR.test(sess, args, bicubic_LR_y)
                else:
                    SR_y = SR.test(sess, args, LR_y)
                SR_y = np.reshape(SR_y, [SR_y.shape[1], SR_y.shape[2], 1])
                bicubic_LR_cb = np.reshape(bicubic_LR_cb, [bicubic_LR_cb.shape[0], bicubic_LR_cb.shape[1], 1])
                bicubic_LR_cr = np.reshape(bicubic_LR_cr, [bicubic_LR_cr.shape[0], bicubic_LR_cr.shape[1], 1])
                SR_ycbcr = np.concatenate((SR_y, bicubic_LR_cb, bicubic_LR_cr),2)
                SR_rgb = ycbcr2rgb(SR_ycbcr)
                imsave(os.path.join(result_path1, os.path.basename(file)), SR_rgb)
            for file in file_list2:
                HR_rgb = Image.open(file)
                LR_rgb = np.clip(imresize(HR_rgb, args.scale, interp='bicubic'),0,255.0)

                LR_ycbcr = rgb2ycbcr(LR_rgb)
                LR_y = LR_ycbcr[:,:,0]
                bicubic_LR_cb = np.clip(imresize(LR_ycbcr[:,:,1], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_cr = np.clip(imresize(LR_ycbcr[:,:,2], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_y = np.clip(imresize(LR_y, 1/args.scale, interp='bicubic'),0,255.0)
                if args.model is not 'DSRN':
                    SR_y = SR.test(sess, args, bicubic_LR_y)
                else:
                    SR_y = SR.test(sess, args, LR_y)
                SR_y = np.reshape(SR_y, [SR_y.shape[1], SR_y.shape[2], 1])
                bicubic_LR_cb = np.reshape(bicubic_LR_cb, [bicubic_LR_cb.shape[0], bicubic_LR_cb.shape[1], 1])
                bicubic_LR_cr = np.reshape(bicubic_LR_cr, [bicubic_LR_cr.shape[0], bicubic_LR_cr.shape[1], 1])
                SR_ycbcr = np.concatenate((SR_y, bicubic_LR_cb, bicubic_LR_cr),2)
                SR_rgb = ycbcr2rgb(SR_ycbcr)
                imsave(os.path.join(result_path2, os.path.basename(file)), SR_rgb)
            for file in file_list3:
                HR_rgb = Image.open(file)
                LR_rgb = np.clip(imresize(HR_rgb, args.scale, interp='bicubic'),0,255.0)

                LR_ycbcr = rgb2ycbcr(LR_rgb)
                LR_y = LR_ycbcr[:,:,0]
                bicubic_LR_cb = np.clip(imresize(LR_ycbcr[:,:,1], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_cr = np.clip(imresize(LR_ycbcr[:,:,2], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_y = np.clip(imresize(LR_y, 1/args.scale, interp='bicubic'),0,255.0)
                if args.model is not 'DSRN':
                    SR_y = SR.test(sess, args, bicubic_LR_y)
                else:
                    SR_y = SR.test(sess, args, LR_y)
                SR_y = np.reshape(SR_y, [SR_y.shape[1], SR_y.shape[2], 1])
                bicubic_LR_cb = np.reshape(bicubic_LR_cb, [bicubic_LR_cb.shape[0], bicubic_LR_cb.shape[1], 1])
                bicubic_LR_cr = np.reshape(bicubic_LR_cr, [bicubic_LR_cr.shape[0], bicubic_LR_cr.shape[1], 1])
                SR_ycbcr = np.concatenate((SR_y, bicubic_LR_cb, bicubic_LR_cr),2)
                SR_rgb = ycbcr2rgb(SR_ycbcr)
                imsave(os.path.join(result_path3, os.path.basename(file)), SR_rgb)
            for file in file_list4:
                HR_rgb = Image.open(file)
                LR_rgb = np.clip(imresize(HR_rgb, args.scale, interp='bicubic'),0,255.0)

                LR_ycbcr = rgb2ycbcr(LR_rgb)
                LR_y = LR_ycbcr[:,:,0]
                bicubic_LR_cb = np.clip(imresize(LR_ycbcr[:,:,1], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_cr = np.clip(imresize(LR_ycbcr[:,:,2], 1/(args.scale), interp='bicubic'),0,255.0)
                bicubic_LR_y = np.clip(imresize(LR_y, 1/args.scale, interp='bicubic'),0,255.0)
                if args.model is not 'DSRN':
                    SR_y = SR.test(sess, args, bicubic_LR_y)
                else:
                    SR_y = SR.test(sess, args, LR_y)
                SR_y = np.reshape(SR_y, [SR_y.shape[1], SR_y.shape[2], 1])
                bicubic_LR_cb = np.reshape(bicubic_LR_cb, [bicubic_LR_cb.shape[0], bicubic_LR_cb.shape[1], 1])
                bicubic_LR_cr = np.reshape(bicubic_LR_cr, [bicubic_LR_cr.shape[0], bicubic_LR_cr.shape[1], 1])
                SR_ycbcr = np.concatenate((SR_y, bicubic_LR_cb, bicubic_LR_cr),2)
                SR_rgb = ycbcr2rgb(SR_ycbcr)
                imsave(os.path.join(result_path4, os.path.basename(file)), SR_rgb)
if __name__=='__main__':
    main()
