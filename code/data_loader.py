import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

def data_loader(HR_path, scale):
    HR = np.load(HR_path)
    LR = np.zeros([HR.shape[0], int(HR.shape[1] * scale), int(HR.shape[2]*scale)])
    bicubic_LR = np.zeros(HR.shape)
    for i in range(HR.shape[0]):
        LR[i,:,:] = np.clip(imresize(HR[i,:,:], scale, interp='bicubic'),0,255.0)
        HR[i,:,:] = HR[i,:,:]
        LR[i,:,:] = LR[i,:,:]
        bicubic_LR[i,:,:] = np.clip(imresize(LR[i,:,:], 1/scale, interp='bicubic'),0,255.0)
        bicubic_LR[i,:,:] = bicubic_LR[i,:,:]
    HR = np.reshape(HR, [-1, 128, 128, 1])
    bicubic_LR = np.reshape(bicubic_LR, [-1, 128, 128, 1])
    LR = np.reshape(LR, [-1,int(128*scale), int(128*scale), 1])
    return HR, LR, bicubic_LR

if __name__ == '__main__':
    HR_path = '../../data/patch/train/train_HR.npy'
    HR,LR = data_loader(HR_path, 0.5)
    plt.imshow(HR[0,:,:])
    plt.show()
    plt.imshow(LR[0,:,:])
    plt.show()
