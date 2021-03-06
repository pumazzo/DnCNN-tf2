import argparse
import datetime
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.losses import MeanSquaredError
from shared_utils import gaussian_noise_layer,add_noise

from dncnn import DnCNN
from dncnnrn import DnCNNRN
print("loaded modules")
# %%
parser = argparse.ArgumentParser(description='DnCNN tf2')
parser.add_argument('--model', default='DnCNN', choices=['DnCNN', 'DnCNNRN'], type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--train_data', default='/home/andrea/denoiser/BSR/BSDS500/data/images/train', type=str, help='path of train data')
parser.add_argument('--test_data', default='/home/andrea/denoiser/BSR/BSDS500/data/images/test', type=str, help='path of test data')
parser.add_argument('--sigma', default=70, type=int, help='noise level (max level for blind mode)')
parser.add_argument('--form', default='GAUSS', choices=['GAUSS', 'RICE'], type=str, help='choose a noise form')
parser.add_argument('--blind', default=True, type=bool, help='blind denoising')
parser.add_argument('--epochs', default=500, type=int, help='number of train epochs')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--train_patch', default=48, type=int, help='size for training patches')
parser.add_argument('--test_size', default=180, type=int, help='size for test images')
parser.add_argument('--format', default='jpg', choices=['jpg', 'png'], type=str, help='image format')
parser.add_argument('--weights_path', default='weights/gauss_20_test', type=str, help='path for saving model weights')
parser.add_argument('--model_path', default='saved_models/gauss_20_test', type=str, help='path for saving whole model')
parser.add_argument('--exp_name', default='gauss_test', type=str, help='name for experiment logs')

args = parser.parse_args()
print("parsed options")
# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE  # for dataset configuration

# Training variables
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd

# Network parameters
MODEL = args.model
DEPTH = args.depth

# Data preparation variables
NOISE_STD = args.sigma
BLIND_MODE = args.blind
NOISE_FORM = args.form
SCALES = [1, 0.9, 0.8, 0.7]  # used for data augmentation
TRAIN_PATCH_DIM = args.train_patch
TEST_DIM = args.test_size
FORMAT = args.format

# Train and test set directories
TEST_DIR = args.test_data + '/*.' + FORMAT
TRAIN_DIR = args.train_data + '/*' + FORMAT

# Paths for saving weights and model
WEIGHTS_PATH = args.weights_path
MODEL_PATH = args.model_path

# Tensorboard logs name
EXPERIMENT_NAME = args.exp_name
print("set dirs")
# %%
# def gaussian_noise_layer(dim):
#     '''generate noise mask of given dimension'''
#     std = NOISE_STD  # random.randint(0, 55) for blind denoising
#     if BLIND_MODE:
#         std = random.randint(0, NOISE_STD) #for blind denoising
#     noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
#     return noise


def augment(image):
    '''prepare and augment input image, and generate noise mask'''
    image = tf.io.read_file(image)

    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)

    # augmentation 1:
    # rescale input to obtain crops at different level of detail
    h, w = float(tf.shape(image)[0]), float(tf.shape(image)[1])
    s = random.choice(SCALES)
    image = tf.image.resize(image, [int(h * s), int(w * s)]) / 250

    # crop random patch
    image = tf.image.random_crop(image, size=[TRAIN_PATCH_DIM, TRAIN_PATCH_DIM, 1])

    # augmentation 2: random flip
    image = tf.image.random_flip_left_right(image)

    # augmentation 3: random rotation (0, 90, 180 or 270 degrees)
    for i in range(np.random.randint(4)):
        image = tf.image.rot90(image)

    # # generate noise mask
    # noise = gaussian_noise_layer(TRAIN_PATCH_DIM)

    # # sum image and noise, clip values between 0 and 1
    # noisy_image = tf.clip_by_value(image + noise, 0, 1)
    
    
    std = NOISE_STD  # random.randint(0, 55) for blind denoising
    if BLIND_MODE:
        std = random.randint(25, NOISE_STD) #for blind denoising
    
    noisy_image = add_noise(image,TRAIN_PATCH_DIM,NOISE_FORM,std)


    return noisy_image, image


def configure_for_train(ds):
    '''configure loading and batching for training set'''
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def augment_test(image):
    '''load test image'''

    # No augmentation and different size from training
    image = tf.io.read_file(image)

    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)
    
    image = tf.image.resize_with_crop_or_pad(image, TEST_DIM, TEST_DIM)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    std = NOISE_STD  # random.randint(0, 55) for blind denoising
    if BLIND_MODE:
        std = random.randint(35, NOISE_STD) #for blind denoising
    noisy_image = add_noise(image,TEST_DIM,NOISE_FORM,std)

    return noisy_image, image


def configure_for_test(ds):
    '''configure loading and batching for training set'''

    # No random shuffle
    ds = ds.cache()
    ds = ds.map(augment_test, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


@tf.function
def train_step(images, targets):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(targets, predictions)
        predictions = tf.clip_by_value(predictions, 0, 1)
        metric_psnr = tf.image.psnr(predictions, targets, max_val=1.0)
        metric = patch_mean_train(predictions, targets)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_metric_mean_patch(metric)
    train_metric_psnr(metric_psnr)
    #optimizer._decayed_lr(tf.float32)
   # with summary_writer.as_default():  NON FUNGE
    #    tf.summary.scalar('learning_rate', optimizer._decayed_lr(tf.float32), step=train_step_count)


@tf.function
def test_step(images, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(targets, predictions)
    predictions = tf.clip_by_value(predictions, 0, 1)
    t_metric_psnr = tf.image.psnr(predictions, targets, max_val=1.0)
    t_metric = patch_mean_train(predictions, targets)

    test_loss(t_loss)
    test_metric_mean_patch(t_metric)
    test_metric_psnr(t_metric_psnr)


@tf.function
def loss(model,x,y,training):
    # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

def patch_mean_train(y_true, y_pred):
    #tf.math.square(tf.math.reduce_mean(y_true)-tf.math.reduce_mean(y_true))
    #mean of a patch 
    return tf.math.square(tf.math.reduce_mean(y_true,[1,2])-tf.math.reduce_mean(y_pred,[1,2]))


# Load train and test set
train_ds = tf.data.Dataset.list_files(TRAIN_DIR)
train_ds = configure_for_train(train_ds)
test_ds = tf.data.Dataset.list_files(TEST_DIR)
test_ds = configure_for_test(test_ds)
print("load data")
# define model, loss function and optimizer
if MODEL == 'DnCNN':
    model = DnCNN(depth=DEPTH)
elif MODEL == 'DnCNNRN':
    model = DnCNNRN(depth=DEPTH)

loss_object = MeanSquaredError()
# second argument is a list not an array!
schedule_lr = tf.optimizers.schedules.PiecewiseConstantDecay(list(map(lambda x : EPOCHS*x,[7,11])), list(map(lambda x : LEARNING_RATE*x,[1e-0, 1e-1, 1e-2])))# second argument is a list not an array!
schedule_w =  tf.optimizers.schedules.PiecewiseConstantDecay(list(map(lambda x : EPOCHS*x,[7,11])), list(map(lambda x : WEIGHT_DECAY *x,[1e-0, 1e-1, 1e-2])))
#schedule_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 1300, 1e-5, 2)
#schedule_w=tf.optimizers.schedules.PolynomialDecay(1e-4, 1300, 1e-6, 2)
# lr and wd can be a function or a tensor
# lr = LEARNING_RATE * schedule(epoch)
# wd = lambda: WEIGHT_DECAY * schedule(epoch)

# # ...

# optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

#optimizer = tfa.optimizers.AdamW(weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE)

# these objects keep track of losses and metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric_mean_patch = tf.keras.metrics.Mean(name='train_mean_patch')
train_metric_psnr = tf.keras.metrics.Mean(name='train_psnr')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_metric_mean_patch = tf.keras.metrics.Mean(name='test_mean_patch')
test_metric_psnr = tf.keras.metrics.Mean(name='test_psnr')

# Set tensorflow dir for the experiment
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + '_' + EXPERIMENT_NAME
summary_writer = tf.summary.create_file_writer(log_dir)
train_step_count = 0

#lr = LEARNING_RATE * schedule(epoch)
#wd = lambda: WEIGHT_DECAY * schedule(epoch)
#lr = LEARNING_RATE * schedule(step)
#wd = lambda: WEIGHT_DECAY * schedule(step)
optimizer = tfa.optimizers.AdamW(learning_rate=schedule_lr, weight_decay=schedule_w)

for epoch in range(EPOCHS):
    #step = tf.Variable(epoch, trainable=False)
    #lr = LEARNING_RATE * schedule(step)
    #wd = lambda: WEIGHT_DECAY * schedule(step)
    #step=step+1;

# ...

   # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    print("On epoch %d / %d" % (epoch,EPOCHS))
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_metric_mean_patch.reset_states()
    train_metric_psnr.reset_states()
    test_loss.reset_states()
    test_metric_psnr.reset_states()
    test_metric_mean_patch.reset_states()


    # training loop
    for images, targets in train_ds:
        train_step(images, targets)

    # test loop
    for test_images, test_targets in test_ds:
        test_step(test_images, test_targets)

    # log losses and metrics
    
    with summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_patch_mean', train_metric_mean_patch.result(), step=epoch)
        tf.summary.scalar('train_psnr', train_metric_psnr.result(), step=epoch)
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('test_patch_mean', test_metric_mean_patch.result(), step=epoch)
        tf.summary.scalar('test_psnr', test_metric_psnr.result(), step=epoch)
        tf.summary.scalar('learning_rate', optimizer._decayed_lr(tf.float32),step=epoch)

# need weights to load the model for inference
model.save_weights(WEIGHTS_PATH)



# need whole model for post-training quantization
model.save(MODEL_PATH)
#save config file
import json

with open(MODEL_PATH+'/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
