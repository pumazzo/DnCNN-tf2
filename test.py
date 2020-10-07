import argparse
import os

import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.restoration import unsupervised_wiener

from dncnn import DnCNN
from dncnnrn import DnCNNRN
from shared_utils import gaussian_noise_layer,add_noise

parser = argparse.ArgumentParser(description='DnCNN tf2 test')
parser.add_argument('--model', default='DnCNN', choices=['DnCNN', 'DnCNNRN'], type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/set12', type=str, help='path of test data')
parser.add_argument('--sigma', default=35, type=int, help='noise level')
parser.add_argument('--form', default='RICE', choices=['GAUSS', 'RICE'], type=str, help='choose a noise form')
parser.add_argument('--depth', default=20, type=int, help='depth of the model')
parser.add_argument('--test_size', default=180, type=int, help='size for test images')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--format', default='png', choices=['jpg', 'png'], type=str, help='image format')
parser.add_argument('--weights_path', default='weights/rice_20_test', type=str, help='path for loading model weights')
parser.add_argument('--save_plots', action='store_true', help='save plots in plots_dir')
parser.add_argument('--plots_dir', default='plots/rice', type=str, help='path for saving plots')

args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE  # for dataset configuration
#TODO: load from train 
# with open(MODEL_PATH+'/commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)

# print(args)
# Network parameters
MODEL = args.model
DEPTH = args.depth

# Data preparation variables
NOISE_STD = args.sigma
NOISE_FORM = args.form
TEST_DIM = args.test_size
FORMAT = args.format
BATCH_SIZE = args.batch_size

# Train and test set directories
TEST_DIR = args.data_dir + '/*.' + FORMAT

# Paths for saving weights and plots
WEIGHTS_PATH = args.weights_path
SAVE_PLOTS = True#args.save_plots
STORE_AVERAGE = False
PLOTS_DIR = args.plots_dir

if MODEL == 'DnCNN':
    model = DnCNN(depth=DEPTH)
elif MODEL == 'DnCNNRN':
    model = DnCNNRN(depth=DEPTH)

model.load_weights(WEIGHTS_PATH)


# def gaussian_noise_layer(dim):
#     '''generate noise mask of given dimension'''
#     std = NOISE_STD
#     noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
#     return noise


def augment(image,std):
    image = tf.io.read_file(image)
    if FORMAT == 'jpg':
        image = tf.image.decode_jpeg(image, channels=1)
    elif FORMAT == 'png':
        image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_with_crop_or_pad(image, TEST_DIM, TEST_DIM)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    #std = NOISE_STD  # random.randint(0, 55) for blind denoising
    print('test noised with std = %d' % std)
    noisy_image = add_noise(image,TEST_DIM,NOISE_FORM,std)
   
    return noisy_image, image


# def configure_ds(ds):
#     ds = ds.cache()
#     ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
#     ds = ds.batch(BATCH_SIZE)
#     ds = ds.prefetch(buffer_size=AUTOTUNE)
#     return ds
def configure_ds(ds,std):
    ds = ds.cache()
    ds = ds.map(lambda x: augment(x, std), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


@tf.function
def test(images, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    predictions = tf.clip_by_value(predictions, 0, 1)
    t_metric = tf.image.psnr(predictions, targets, max_val=1.0)
    
    
    test_metric(t_metric)


test_metric = tf.keras.metrics.Mean(name='test_metric')
test_metric.reset_states()
test_ds_list = tf.data.Dataset.list_files(TEST_DIR)
test_ds = configure_ds(test_ds_list,NOISE_STD)

for test_images, test_targets in test_ds:
    test(test_images, test_targets)

print(f'Avreage PSNR: {test_metric.result().numpy()}')

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# %%
if SAVE_PLOTS:
    # Plot the results
    # from left to right:
    # noisy image, prediction, true image
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    image_batch, target_batch = next(iter(test_ds))
    predictions = model(image_batch, training=False)
    predictions = tf.clip_by_value(predictions, 0, 1)
    test_ds_list = list(test_ds_list)

    for i in range(len(test_ds_list)):
        orig=target_batch[i].numpy().squeeze()
        noisy=image_batch[i].numpy().squeeze()
        processed=predictions[i].numpy().squeeze()
        wien, _ = unsupervised_wiener(noisy, np.ones((3, 3)) / 9)
        psnr_noise = psnr(noisy, orig)
        psnr_processed = psnr(processed, orig)
        psnr_wien = psnr(wien, orig)
        f, axarr = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        plt.sca(axarr[0,0])
        plt.imshow(noisy, cmap='gray')
        plt.axis("off")
        plt.title('Noisy - PSNR : %2.2f ' %psnr_noise )
        plt.sca(axarr[0,1])
        plt.imshow(processed, cmap='gray')
        plt.axis("off")
        plt.title('Prediction - PSNR : %2.2f ' %psnr_processed )
        plt.sca(axarr[1,1])
        plt.imshow(orig, cmap='gray')
        plt.axis("off")
        plt.title('Original')
        plt.sca(axarr[1,0])
        plt.imshow(wien, cmap='gray')
        plt.axis("off")
        plt.title('baseline - PSNR : %2.2f ' %psnr_wien )
        plt.savefig(f'{PLOTS_DIR}/img_{NOISE_FORM}_{NOISE_STD}_{i + 1}.png')

# import pickle
# d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }
# afile = open(r'C:\d.pkl', 'wb')
# pickle.dump(d, afile)
# afile.close()

# #reload object from file
# file2 = open(r'C:\d.pkl', 'rb')
# new_d = pickle.load(file2)
# file2.close()

# #print dictionary object loaded from file
# print new_d


if STORE_AVERAGE:
    std_array=[]
    noisy_arr=[]
    base_arr=[]
    net_arr=[]
    psnr_noise = []
    psnr_processed = []
    psnr_wien = []
    scatter_x_ypred=np.zeros([len(range(35,60)), 12,2  ])
    scatter_wiener_ypred=np.zeros([len(range(35,60)), 12,2  ])
    
    for idx, j in enumerate(range(35,60)):
        NOISE_STD = j
        std_array.append(NOISE_STD)
        test_ds = configure_ds(test_ds_list,NOISE_STD)
        image_batch, target_batch = next(iter(test_ds))
        predictions = model(image_batch, training=False)
        predictions = tf.clip_by_value(predictions, 0, 1)
        test_list = list(test_ds_list)
    
        for i in range(len(test_list)):
            orig=target_batch[i].numpy().squeeze()
            noisy=image_batch[i].numpy().squeeze()
            processed=predictions[i].numpy().squeeze()
            wien, _ = unsupervised_wiener(noisy, np.ones((3, 3)) / 9)
            psnr_noise.append(psnr(noisy, orig))
            psnr_processed.append(psnr(processed, orig))
            psnr_wien.append(psnr(wien, orig))
            scatter_x_ypred[idx,i,]=[psnr(noisy, orig),psnr(processed, orig)]
            scatter_wiener_ypred[idx,i,]=[psnr(wien, orig),psnr(processed, orig)]
        noisy_arr.append(np.mean(psnr_noise))
        base_arr.append(np.mean(psnr_wien))
        net_arr.append(np.mean(psnr_processed))
        
        
    #plotting
# %%
if STORE_AVERAGE:
    
    #np.random.seed(1974)
    
    # Generate Data
   
    x=np.ndarray.flatten( scatter_x_ypred[:,:,0])
    y=np.ndarray.flatten( scatter_x_ypred[:,:,1])
    labels = np.ndarray.flatten(np.array([range(12),]*25).transpose())
    df = pd.DataFrame(dict(x=x, y=y, label=labels))
    
    groups = df.groupby('label')
    
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    #ax.legend()
    
    #plt.show()    
    # create the line
    
    ax.plot([12, 30],[12, 30], 'r--', label='Random guess')
   
    # add labels, legend and make it nicer
    ax.set_xlabel('noisy')
    ax.set_ylabel('processed')
    ax.set_title('PSNR scatter noisy vs. processed')
    ax.set_xlim(19, 12)
    ax.set_ylim(13, 28)
   #ax.legend()
    plt.tight_layout()
    #plt.show()  
    plt.savefig(f'{PLOTS_DIR}/graph/noise_processed_{NOISE_FORM}_{NOISE_STD}.png')
# %%   
if STORE_AVERAGE:
    
    #np.random.seed(1974)
    
    # Generate Data
   
    x=np.ndarray.flatten( scatter_wiener_ypred[:,:,0])
    y=np.ndarray.flatten( scatter_wiener_ypred[:,:,1])
    labels = np.ndarray.flatten(np.array([range(12),]*25).transpose())
    df = pd.DataFrame(dict(x=x, y=y, label=labels))
    
    groups = df.groupby('label')
    
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    #ax.legend()
    
    #plt.show()    
    # create the line
    
    ax.plot([12, 30],[12, 30], 'r--', label='Random guess')
   
    # add labels, legend and make it nicer
    ax.set_xlabel('wiener (baseline)')
    ax.set_ylabel('processed')
    ax.set_title('PSNR scatter weiner vs.processed')
    ax.set_xlim(14, 25)
    ax.set_ylim(13, 28)
   #ax.legend()
    plt.tight_layout()
    #plt.show()  
    plt.savefig(f'{PLOTS_DIR}/graph/wiener_processed_{NOISE_FORM}_{NOISE_STD}.png')
    
# %%
if STORE_AVERAGE:
    plt.close()
    x = std_array
    y1 = net_arr
    y2=base_arr
    plt.plot(x, y1,'-o',label='CNN')
    plt.plot(x, y2,'-o',label='WEINER')
    plt.legend()
    plt.xlabel('STD NOISE')
    plt.ylabel('PSNR')
    plt.title(f'BLIND TRAINING\n{NOISE_FORM}')
    plt.grid(True)
    plt.savefig(f'{PLOTS_DIR}/graph/denoising_{NOISE_FORM}_{NOISE_STD}.png')
    