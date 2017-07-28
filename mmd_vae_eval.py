import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math, os, time
from tensorflow.examples.tutorials.mnist import input_data
import subprocess
import argparse
from scipy import misc as misc

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-r', '--reg_type', type=str, default='mmd', help='Type of regularization')
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('-n', '--train_size', type=int, default=50000, help='Number of samples for training')
args = parser.parse_args()


reg_type = args.reg_type
train_size = args.train_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 400


def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path


log_path = make_model_path('%s_%d' % (reg_type, train_size))


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


# Encoder and decoder use the DC-GAN architecture
# 28 x 28 x 1
def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)   # None x 7 x 7 x 128
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.005)
        return mean, stddev


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        mean = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        stddev = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.005)
        return mean, stddev


# Build the computation graph for training
z_dim = 5
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None]+x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
train_xmean, train_xstddev = decoder(train_z)
train_xr = train_xmean + tf.multiply(train_xstddev,
                                     tf.random_normal(tf.stack([tf.shape(train_x)[0]] + x_dim)))

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean, gen_xstddev = decoder(gen_z, reuse=True)

sample_nll = tf.div(tf.square(train_x - gen_xmean), tf.square(gen_xstddev)) / 2.0 + tf.log(gen_xstddev)
sample_nll += math.log(2 * np.pi) / 2.0
sample_nll = tf.reduce_sum(sample_nll, axis=(1, 2, 3))     # negative log likelihood per dimension


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z)

# ELBO loss divided by input dimensions
loss_elbo = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                          0.5 * tf.square(train_zmean) - 0.5, axis=1)
loss_elbo = tf.reduce_mean(loss_elbo) / np.prod(x_dim)

# Negative log likelihood per dimension
loss_nll = tf.div(tf.square(train_x - train_xmean), tf.square(train_xstddev)) / 2.0 + tf.log(train_xstddev)
loss_nll = tf.reduce_mean(loss_nll)
loss_nll += math.log(2 * np.pi) / 2.0

if reg_type == 'mmd':
    loss = loss_nll + 50 * loss_mmd
elif reg_type == 'elbo':
    loss = loss_nll + loss_elbo
else:
    print("Unknown type")
    exit(-1)

trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)
logger = open(os.path.join(log_path, 'log.txt'), 'w')


# Train on limited data
class LimitedMnist:
    def __init__(self, mnist, size):
        self.data_ptr = 0
        self.size = size
        assert size <= mnist.train.images.shape[0]
        self.data = mnist.train.images[:size]

    def next_batch(self, batch_size):
        prev_ptr = self.data_ptr
        self.data_ptr += batch_size
        if self.data_ptr > self.size:
            prev_ptr = 0
            self.data_ptr = batch_size
        return self.data[prev_ptr:self.data_ptr]


mnist = input_data.read_data_sets('mnist_data')
limited_mnist = LimitedMnist(mnist, train_size)


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, max_samples=100):
    if max_samples > samples.shape[0]:
        max_samples = samples.shape[0]
    cnt, height, width = int(math.floor(math.sqrt(max_samples))), samples.shape[1], samples.shape[2]
    samples = samples[:cnt*cnt]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

# Start training
# plt.ion()
for i in range(100000):
    batch_x = limited_mnist.next_batch(batch_size)
    batch_x = np.reshape(batch_x, [-1] + x_dim)
    _, nll, mmd, elbo, xmean, xstddev = sess.run([trainer, loss_nll, loss_mmd, loss_elbo, train_xmean, train_xstddev], feed_dict={train_x: batch_x})
    if i % 100 == 0:
        print("Iteration %d, nll %.4f, mmd loss %.4f, elbo loss %.4f" % (i, nll, mmd, elbo))
    if i % 2000 == 0:
        samples, sample_stddev = sess.run([gen_xmean, gen_xstddev], feed_dict={gen_z: np.random.normal(size=(100, z_dim))})
        plots = np.stack([convert_to_display(samples), convert_to_display(sample_stddev),
                          convert_to_display(xmean), convert_to_display(xstddev)], axis=0)
        plots = np.expand_dims(plots, axis=-1)
        plots = convert_to_display(plots)
        misc.imsave(os.path.join(log_path, 'samples%d.png' % i), plots)


def compute_log_sum(val):
    min_val = np.min(val, axis=0, keepdims=True)
    return np.mean(min_val - np.log(np.mean(np.exp(-val + min_val), axis=0)))


print("---------------------> Computing true log likelihood")
start_time = time.time()
train_avg_nll = []
test_avg_nll = []
for i in range(50):
    if i % 2 == 0:
        batch_x = limited_mnist.next_batch(batch_size)
        run_name = '%s-%d-train' % (reg_type, train_size)
    else:
        batch_x, _ = mnist.test.next_batch(batch_size)
        run_name = '%s-%d-test' % (reg_type, train_size)
    batch_x = np.reshape(batch_x, [-1] + x_dim)
    nll_list = []
    num_iter = 50000
    for k in range(num_iter):
        random_z = np.random.normal(size=[batch_size, z_dim])
        nll = sess.run(sample_nll, feed_dict={train_x: batch_x, gen_z: random_z})
        nll_list.append(nll)
        if k % 20000 == 0:
            print("%s: iter %d, current value %.4f, time used %.2f" % (run_name, k, compute_log_sum(np.stack(nll_list)), time.time() - start_time))
    nll = compute_log_sum(np.stack(nll_list))
    print("%s likelihood importance sampled = %.4f, time used %.2f" % (run_name, nll, time.time() - start_time))

    if i % 2 == 0:
        train_avg_nll.append(nll)
        logger.write('train %f\n' % nll)
    else:
        test_avg_nll.append(nll)
        logger.write('test %f\n' % nll)
    logger.flush()
train_nll = np.mean(train_avg_nll)
test_nll = np.mean(test_avg_nll)
print("Estimated log likelihood is train %f/test %f, time elapsed %f" % (train_nll, test_nll, time.time() - start_time))