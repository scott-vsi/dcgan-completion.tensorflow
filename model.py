# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange
import scipy.misc

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, 3]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = 3 # FIXME should be c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        #self.g_bn4 = batch_norm(name='g_bn4')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        self.z2 = tf.nn.l2_normalize(tf.random_normal(shape=(self.batch_size*2, self.z_dim), mean=0.0, stddev=1.0), dim=1)
        self.z2_sum = tf.histogram_summary("z2", self.z2)
        self.z = self.z2[:self.batch_size, ...]
        self.z_sum = tf.histogram_summary("z", self.z)

        sample_z = np.random.normal(0, 1, size=[self.sample_size, self.z_dim]) \
                     .astype(np.float32)
        sample_z /= np.expand_dims(np.linalg.norm(sample_z, axis=1, ord=2), axis=1)
        self.sample_z = tf.get_variable('sample_z', initializer=sample_z)

        self.G = self.generator(self.z)
        self.G2 = tf.concat(0, [
            self.generator(self.z2[:self.batch_size,...], should_reuse=True), # == self.G
            self.generator(self.z2[self.batch_size:,...], should_reuse=True)])
        self.D_real, self.D_logits_real = self.discriminator(self.images)
        self.D_fake, self.D_logits_fake = self.discriminator(self.G, should_reuse=True)
        # ugh so much repetition
        self.D2_fake, self.D2_logits_fake = [tf.concat(0, [l0, l1]) for l0,l1 in 
            zip(self.discriminator(self.G2[:self.batch_size,...], should_reuse=True), 
                self.discriminator(self.G2[self.batch_size:,...], should_reuse=True))]

        self.sampler = self.generator(self.sample_z, should_reuse=True, is_train=False)
        self.D_sample, self.D_logits_sample = self.discriminator(self.sampler, should_reuse=True, is_train=False)
        # FIXME this should be run after self.sample_z but before self.contextual_loss
        self.normalize_sample_z_op = tf.assign(self.sample_z, tf.nn.l2_normalize(self.sample_z, dim=1))

        self.D_real_sum = tf.histogram_summary("D_real", self.D_real)
        self.D_fake_sum = tf.histogram_summary("D_fake", self.D_fake)
        self.D2_fake_sum = tf.histogram_summary("D2_fake", self.D2_fake)
        self.G_sum = tf.image_summary("G", self.G)
        self.G2_sum = tf.image_summary("G2", self.G2)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_real,
                                                    tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake,
                                                    tf.zeros_like(self.D_fake)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_fake,
                                                    tf.ones_like(self.D2_fake)))

        self.g_loss_sample = tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_sample,
                                                    tf.ones_like(self.D_sample))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.sampler) - tf.mul(self.mask, self.images))), 1) / \
            tf.reduce_sum(tf.contrib.layers.flatten(self.mask), 1)
        self.perceptual_loss = self.g_loss_sample
        self.complete_loss = (1.0-self.lam)*self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z) # FIXME should this be self.sample_z

    def train(self, config):
        data = glob(os.path.join(config.dataset, "*/*.JPEG"))
        np.random.shuffle(data)
        assert(len(data) > 0)

        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars, global_step=global_step)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.z2_sum, self.D2_fake_sum, self.G2_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.D_real_sum, self.D_fake_sum, self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's Brandon Amos'
trained model for faces that's used in the post.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")

        for epoch in xrange(config.epoch):
            #data = glob(os.path.join(config.dataset, "*/*.JPEG"))
            np.random.shuffle(data)
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, errD_fake, errD_real, summary_str = self.sess.run([d_optim, self.d_loss_fake, self.d_loss_real, self.d_sum],
                    feed_dict={ self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                # ensure generator sees as many images as the discriminator
                # REVIEW running g_optim twice, as in the original implementation, may accomplish essentially the same thing
                _, errG, summary_str = self.sess.run([g_optim, self.g_loss, self.g_sum])
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def complete(self, config):
        if not os.path.exists(os.path.join(config.outDir, 'hats_imgs')):
            os.makedirs(os.path.join(config.outDir, 'hats_imgs'))
        if not os.path.exists(os.path.join(config.outDir, 'completed')):
            os.makedirs(os.path.join(config.outDir, 'completed'))

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        elif config.maskType == 'image': # [0,1] 0 is unknown
            # deal with resizing interpolation
            im = scipy.misc.imread(config.mask, mode='RGB')
            cropped = center_crop(im) if self.is_crop else im
            # resize works better if the dtype is not float
            resized = scipy.misc.imresize(cropped, [self.image_size, self.image_size], interp='nearest')
            # rescale byte encoded images
            mask = resized / 255.0 if np.max(resized) > 1 else resized
        else:
            assert(False)

        optim = tf.train.AdamOptimizer(config.lr, beta1=config.momentum) \
                .minimize(self.complete_loss, var_list=[self.sample_z], grad_loss=self.grad_complete_loss[0])

        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = 8
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))

            for i in xrange(config.nIter):
                fd = {
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                (_, sample_z, G_imgs, 
                    loss, contextual_loss, perceptual_loss, D_logits_sample) = self.sess.run([
                        optim, self.normalize_sample_z_op, self.sampler, 
                        self.complete_loss, self.contextual_loss, self.perceptual_loss, self.D_logits_sample], 
                        feed_dict=fd)
                #print "contextual_loss: ", contextual_loss
                #print "perceptual_loss: ", perceptual_loss.T
                #print "D_logits_sample: ", D_logits_sample.T
                #print

                if i % 5 == 0:
                    #avg_loss = np.mean(loss[0:batchSz])
                    #if i == 0: print(i, avg_loss)
                    #else: print(i, avg_loss, prev_avg_loss-avg_loss)
                    #prev_avg_loss = avg_loss

                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows,nCols = np.ceil(batchSz/8),8
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completeed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)


    def discriminator(self, image, should_reuse=False, is_train=True, with_instance_noise=True):
        with tf.variable_scope('discriminator'):
            if should_reuse:
                tf.get_variable_scope().reuse_variables()

            # switch for when doing completion (i think we want this)
            if with_instance_noise:
                # instance noise
                # REVIEW should this be different for each image
                additive_gaussian_noise = tf.random_normal(shape=(self.image_size,self.image_size), mean=0.0, stddev=0.04)
                additive_gaussian_noise = tf.expand_dims(tf.expand_dims(additive_gaussian_noise, dim=0), dim=-1)
                image = tf.add(image, tf.tile(additive_gaussian_noise, multiples=(self.batch_size,1,1,self.c_dim)))
                max_per_image = tf.reduce_max(tf.abs(image), reduction_indices=(1,2,3))
                max_per_image = tf.expand_dims(tf.expand_dims(tf.expand_dims(max_per_image, dim=-1), dim=-1), dim=-1)
                image = tf.div(image, tf.tile(max_per_image, multiples=(1,self.image_size,self.image_size,self.c_dim)))

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), train=is_train))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'), train=is_train))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), train=is_train))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, should_reuse=False, is_train=True):
        with tf.variable_scope('generator'):
            if should_reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=is_train))

            h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=is_train))

            h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=is_train))

            h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=is_train))

            h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h4)

    def generator_128(self, z, should_reuse=False, is_train=True):
        with tf.variable_scope('generator_128'):
            if should_reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = tf.reshape(linear(z, self.gf_dim*16*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 16])
            h0 = tf.nn.relu(self.g_bn0(h0, train=is_train))

            h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*8], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=is_train))

            h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*4], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=is_train))

            h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*2], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=is_train))

            h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, self.gf_dim*1], name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4, train=is_train))

            h5 = conv2d_transpose(h4, [self.batch_size, 128, 128, 3], name='g_h5')

            return tf.nn.tanh(h5)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
