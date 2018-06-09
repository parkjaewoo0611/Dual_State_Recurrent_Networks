import os
from sklearn.utils import shuffle
from datetime import datetime
from scipy.misc import imresize
from tqdm import trange, tqdm
from scipy.misc import imsave, imread
import tensorflow as tf
import numpy as np
from SSRN import SingleState as SSRN
from DSRN import DualState as DSRN
from ops import *
class SuperResolution(object):
    def __init__(self, args):
        self._log_step = 50
        self._batch_size = 8
        self._scale = args.scale
        self._lr = 0.001
        self._is_train = args.is_train
        self.log_dir = os.path.join('logs', args.model + '_' + str(int(1/(args.scale))))
        self.weight_dir = self.log_dir + '/model.ckpt'
        self.result_dir = os.path.join('result', args.model +'_'+ str(int(1/(args.scale))))

        self.summary_writer = tf.summary.FileWriter(self.log_dir)

        self.LR_input = LR_input = tf.placeholder(tf.float32, [None, None, None, 1] ,name='LR_image')
        self.HR_input = HR_input = tf.placeholder(tf.float32, [None, None, None, 1], name='HR_image')
        self.lr_input = lr_input = tf.placeholder(tf.float32, name='learning_rate')

        k1 = conv2d(input=LR_input, name='input_conv1', num_filters=64,  filter_size=3, stride=1, reuse=False)
        k1 = tf.nn.relu(k1)

        k2 = conv2d(input=k1, name='input_conv2', num_filters=128, filter_size=3, stride=1, reuse=False)
        k2 = tf.nn.relu(k2)

        # model
        if args.model == 'ResNet':
            """ single state ResNet """
            with tf.variable_scope('ResNet'):

                s1 = residual(input=k2, name='Res_recur', num_filters=128, reuse=False)
                s1 = prelu(input=s1, name='Res_recur1')

                s2 = residual(input=s1, name='Res_recur', num_filters=128, reuse=True)
                s2 = prelu(input=s2, name='Res_recur2')

                s3 = residual(input=s2, name='Res_recur', num_filters=128, reuse=True)
                s3 = prelu(input=s3, name='Res_recur3')

                s4 = residual(input=s3, name='Res_recur', num_filters=128, reuse=True)
                s4 = prelu(input=s4, name='Res_recur4')

                s5 = residual(input=s4, name='Res_recur', num_filters=128, reuse=True)
                s5 = prelu(input=s5, name='Res_recur5')

                s6 = residual(input=s5, name='Res_recur', num_filters=128, reuse=True)
                s6 = prelu(input=s6, name='Res_recur6')

                s7 = residual(input=s6, name='Res_recur', num_filters=128, reuse=True)
                s7 = prelu(input=s7, name='Res_recur7')

                y = conv2d(input=s7, name='Res_out', num_filters=1, filter_size=3, stride=1, reuse=False)

        elif args.model == 'DRCN':
            """ single state DRCN """
            with tf.variable_scope('DRCN'):

                s1  = conv2d(input=k2,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=False)
                s1 = prelu(input=s1, name='DRCN_recur1')

                s2  = conv2d(input=s1,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s2 = prelu(input=s2, name='DRCN_recur2')

                s3  = conv2d(input=s2,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s3 = prelu(input=s3, name='DRCN_recur3')

                s4  = conv2d(input=s3,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s4 = prelu(input=s4, name='DRCN_recur4')

                s5  = conv2d(input=s4,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s5 = prelu(input=s5, name='DRCN_recur5')

                s6  = conv2d(input=s5,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s6 = prelu(input=s6, name='DRCN_recur6')

                s7  = conv2d(input=s6,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s7 = prelu(input=s7, name='DRCN_recur7')

                s8  = conv2d(input=s7,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s8 = prelu(input=s8, name='DRCN_recur8')

                s9  = conv2d(input=s8,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s9 = prelu(input=s9, name='DRCN_recur9')

                s10 = conv2d(input=s9,  name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s10 = prelu(input=s10, name='DRCN_recur10')

                s11 = conv2d(input=s10, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s11 = prelu(input=s11, name='DRCN_recur11')

                s12 = conv2d(input=s11, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s12 = prelu(input=s12, name='DRCN_recur12')

                s13 = conv2d(input=s12, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s13 = prelu(input=s13, name='DRCN_recur13')

                s14 = conv2d(input=s13, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s14 = prelu(input=s14, name='DRCN_recur14')

                s15 = conv2d(input=s14, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s15 = prelu(input=s15, name='DRCN_recur15')

                s16 = conv2d(input=s15, name='DRCN_recur', num_filters=128, filter_size=3, stride=1, reuse=True)
                s16 = prelu(input=s16, name='DRCN_recur16')

                y0  = conv2d(input=k2,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=False)
                y0 = prelu(input=y0, name='DRCN_out0')

                y1  = conv2d(input=s1,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y1 = prelu(input=y1, name='DRCN_out1')

                y2  = conv2d(input=s2,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y2 = prelu(input=y2, name='DRCN_out2')

                y3  = conv2d(input=s3,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y3 = prelu(input=y3, name='DRCN_out3')

                y4  = conv2d(input=s4,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y4 = prelu(input=y4, name='DRCN_out4')

                y5  = conv2d(input=s5,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y5 = prelu(input=y5, name='DRCN_out5')

                y6  = conv2d(input=s6,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y6 = prelu(input=y6, name='DRCN_out6')

                y7  = conv2d(input=s7,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y7 = prelu(input=y7, name='DRCN_out7')

                y8  = conv2d(input=s8,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y8 = prelu(input=y8, name='DRCN_out8')

                y9  = conv2d(input=s9,  name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y9 = prelu(input=y9, name='DRCN_out9')

                y10 = conv2d(input=s10, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y10 = prelu(input=y10, name='DRCN_out10')

                y11 = conv2d(input=s11, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y11 = prelu(input=y11, name='DRCN_out11')

                y12 = conv2d(input=s12, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y12 = prelu(input=y12, name='DRCN_out12')

                y13 = conv2d(input=s13, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y13 = prelu(input=y13, name='DRCN_out13')

                y14 = conv2d(input=s14, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y14 = prelu(input=y14, name='DRCN_out14')

                y15 = conv2d(input=s15, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y15 = prelu(input=y15, name='DRCN_out15')

                y16 = conv2d(input=s16, name='DRCN_out', num_filters=1, filter_size=3, stride=1, reuse=True)
                y16 = prelu(input=y16, name='DRCN_out16')


                y = tf.scalar_mul(1/17, y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16)

        elif args.model == 'DRRN':
            """ single state DRRN """
            with tf.variable_scope('DRRN'):

                s1 = conv2d(input=k2,  name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=False)
                s1 = prelu(input=s1, name='DRRN_recur_p1')
                s2 = conv2d(input=s1,  name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=False)
                s2 = prelu(input=s2, name='DRRN_recur_p2')
                s2 = s2 + k2

                s3 = conv2d(input=s2,  name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s3 = prelu(input=s3, name='DRRN_recur_p3')
                s4 = conv2d(input=s3,  name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s4 = prelu(input=s4, name='DRRN_recur_p4')
                s4 = s4 + k2

                s5 = conv2d(input=s4,  name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s5 = prelu(input=s5, name='DRRN_recur_p5')
                s6 = conv2d(input=s5,  name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s6 = prelu(input=s6, name='DRRN_recur_p6')
                s6 = s6 + k2

                s7 = conv2d(input=s6, name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s7 = prelu(input=s7, name='DRRN_recur_p7')
                s8 = conv2d(input=s7,name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s8 = prelu(input=s8, name='DRRN_recur_p8')
                s8 = s8 + k2

                s9 = conv2d(input=s8,name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s9 = prelu(input=s9, name='DRRN_recur_p9')
                s10 = conv2d(input=s9,name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s10 = prelu(input=s10, name='DRRN_recur_p10')
                s10 = s10 + k2

                s11 = conv2d(input=s10,name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s11 = prelu(input=s11, name='DRRN_recur_p11')
                s12 = conv2d(input=s11,name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s12 = prelu(input=s12, name='DRRN_recur_p12')
                s12 = s12 + k2

                s13 = conv2d(input=s12,name='DRRN_recur1', num_filters=128, filter_size=3, stride=1, reuse=True)
                s13 = prelu(input=s13, name='DRRN_recur_p13')
                s14 = conv2d(input=s13,name='DRRN_recur2', num_filters=128, filter_size=3, stride=1, reuse=True)
                s14 = prelu(input=s14, name='DRRN_recur_p14')
                s14 = s14 + k2

                y = conv2d(input=s14, name='DRRN_out', num_filters=1, filter_size=3, stride=1, reuse=False)

        elif args.model == 'DSRN':
            """ Dual state DSRN """
            with tf.variable_scope('DSRN'):
                s_HR = conv2d_transpose(input=k2, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=False)
                s_HR = prelu(input=s_HR, name='DSRN_HR_1')
                y = conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=False)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=False)+\
                       residual(input=k2, name='DSRN_f_lr', num_filters=128, reuse=False)
                s_LR = prelu(input=s_LR, name='DSRN_LR_2')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=False)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_2')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)+\
                       residual(input=s_LR, name='DSRN_f_lr', num_filters=128, reuse=True)
                s_LR = prelu(input=s_LR, name='DSRN_LR_3')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=True)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_3')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)+\
                       residual(input=s_LR, name='DSRN_f_lr', num_filters=128, reuse=True)
                s_LR = prelu(input=s_LR, name='DSRN_LR_4')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=True)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_4')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)+\
                       residual(input=s_LR, name='DSRN_f_lr', num_filters=128, reuse=True)
                s_LR = prelu(input=s_LR, name='DSRN_LR_5')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=True)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_5')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)+\
                       residual(input=s_LR, name='DSRN_f_lr', num_filters=128, reuse=True)
                s_LR = prelu(input=s_LR, name='DSRN_LR_6')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=True)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_6')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)

                s_LR = conv2d(input=s_HR, name='DSRN_f_down', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)+\
                       residual(input=s_LR, name='DSRN_f_lr', num_filters=128, reuse=True)
                s_LR = prelu(input=s_LR, name='DSRN_LR_7')
                s_HR = residual(input=s_HR, name='DSRN_f_hr', num_filters=128, reuse=True)+\
                       conv2d_transpose(input=s_LR, name='DSRN_f_up', num_filters=128, filter_size=3, stride=int(1/self._scale), reuse=True)
                S_HR = prelu(input=s_HR, name='DSRN_HR_7')
                y = y + conv2d(input=s_HR, name='DSRN_out', num_filters=1, filter_size=3, stride=1, reuse=True)


                y = tf.scalar_mul(1/7, y)

        # image generation ab, ba, aba, bab
        SR  = self.SR = y
        #  latent losses for each type model
        loss = self.loss = tf.reduce_mean(tf.nn.l2_loss(SR - HR_input))
        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.lr_input)
        gvs = opt.compute_gradients(loss)
        capped_gvs = [(grad, var) if grad is None else (tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
        self.optimizer = opt.apply_gradients(capped_gvs)
        # Summaries
        tf.summary.scalar('model/lr', self.lr_input)
        tf.summary.scalar('loss', self.loss)

        tf.summary.image('image/HR', HR_input[0:1])
        tf.summary.image('image/LR', LR_input[0:1])
        tf.summary.image('image/SR', SR[0:1])

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self, sess, args, data_LR, data_HR, val_data_LR, val_data_HR):
        data_N = len(data_HR)
        total_epoch = 100
        num_iter = data_N // self._batch_size

        sess.run(tf.global_variables_initializer())
        val_iter = int(val_data_HR.shape[0] / self._batch_size)
        for step in range(0, num_iter * total_epoch):
            epoch = step // num_iter
            iter = step % num_iter

            if iter == 0 :
                data_HR, data_LR = shuffle(data_HR, data_LR)
                fetches = [self.SR, self.summary_op]

                HR = np.stack(data_HR[iter*self._batch_size:(iter+1)*self._batch_size])
                LR = np.stack(data_LR[iter*self._batch_size:(iter+1)*self._batch_size])
                SR, summ = sess.run(fetches, feed_dict={self.LR_input: LR,
                                                        self.HR_input: HR,
                                                        self.lr_input: self._lr})

                path_HR = os.path.join(self.result_dir, 'SR_{}_HR.jpg'.format(step))
                path_LR = os.path.join(self.result_dir, 'SR_{}_LR.jpg'.format(step))
                path_SR = os.path.join(self.result_dir, 'SR_{}_SR.jpg'.format(step))
                path = os.path.join(self.result_dir, 'SR_{}_total.jpg'.format(step))
                if args.model == 'DSRN':
                    resized_LR = np.zeros(HR.shape)
                    for i in range(HR.shape[0]):
                        resized_LR[i,:,:,0] = np.clip(imresize(LR[i,:,:,0], 1/args.scale, interp='bicubic'), 0, 255.0)
                    LR = resized_LR

                result_HR = np.clip(HR[0,:,:,0], 0, 255.0)
                result_LR = np.clip(LR[0,:,:,0], 0, 255.0)
                result_SR = np.clip(SR[0,:,:,0], 0, 255.0)
                result = np.concatenate((result_HR, result_LR, result_SR), 1)
                imsave(path_HR, result_HR)
                imsave(path_LR, result_LR)
                imsave(path_SR, result_SR)
                imsave(path, result)
                self.summary_writer.add_summary(summ, step)
                self.summary_writer.flush()
                self.saver.save(sess, self.weight_dir)

                if epoch % 20 == 0:
                    self._lr = self._lr/5
                    if self._lr<1e-7:
                        break

            HR = np.stack(data_HR[iter*self._batch_size:(iter+1)*self._batch_size])
            LR = np.stack(data_LR[iter*self._batch_size:(iter+1)*self._batch_size])

            _, loss=sess.run([self.optimizer, self.loss], feed_dict={self.HR_input: HR,
                                                                     self.LR_input: LR,
                                                                     self.lr_input: self._lr})
            print('step:',str(step),' loss:', str(loss))

    def test(self, sess, args, input_img):
        img = np.reshape(input_img, [1, input_img.shape[0], input_img.shape[1], 1])
        SR = sess.run(self.SR, feed_dict={self.LR_input: img})
        return SR
