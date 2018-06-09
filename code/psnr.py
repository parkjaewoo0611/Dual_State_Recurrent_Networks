import numpy as np
import os
import sys
import math
import matplotlib.image as mpimg

def psnr(imageA, imageB):
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse = mse / float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    psnr = 10 * np.log10(1 / mse)

    return psnr


original_path = sys.argv[1]
result_path = sys.argv[2]

original_list = os.listdir(original_path)
print(original_list)

psnr_ = np.ndarray(shape=[len(original_list)])

kk = 0

for i in range(len(original_list)):
    current = original_list[i]

    original = mpimg.imread(os.path.join(original_path, current))
    result = mpimg.imread(os.path.join(result_path, current))

    m = psnr(original, result)

    print(original_list[i], m)

    psnr_[i] = m

    if m == math.inf:
        psnr_[i] = 0
        kk = kk + 1

print(np.average(psnr_, 0) * 100 / (100.0 - kk), kk)
