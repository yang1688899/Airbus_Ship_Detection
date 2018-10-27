import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import pickle
from skimage.morphology import label

import config


def save_to_pickle(obj,savepath):
    with open(savepath,"wb") as file:
        pickle.dump(obj,file)

def load_pickle(path):
    with open(path,"rb") as file:
        obj = pickle.load(file)
        return obj


def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels == k, axis=2), **kwargs) for k in np.unique(labels[labels > 0])]
    else:
        return [rle_encode(labels == k, **kwargs) for k in np.unique(labels[labels > 0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def load_img(path):
    img = cv2.imread(path)
    if not config.TARGET_SIZE==config.ORIGIN_SIZE:
        img_re = cv2.resize(img,config.TARGET_SIZE)
    return img_re/255.

def generate_mask(rle_list,shape=[768,768]):
    mask_all = np.zeros(shape=shape)
    for rle in rle_list:
        if rle==rle:
            mask_all += rle_decode(rle,shape=shape)
    if not config.TARGET_SIZE == config.ORIGIN_SIZE:
        mask_all = cv2.resize(mask_all,config.TARGET_SIZE)
    return np.expand_dims(mask_all,axis=2)


def data_generator(img_paths,train_tf,batch_size=32, is_shuffle = True):
    num_sample = len(img_paths)
    if is_shuffle:
        img_paths  = shuffle(img_paths)

    while True:
        if is_shuffle:
            img_paths = shuffle(img_paths)
        for offset in range(0,num_sample,batch_size):
            features = [load_img(path) for path in img_paths[offset:offset+batch_size]]
            img_ids = [path.split("/")[-1] for path in img_paths[offset:offset+batch_size]]
            labels = []
            for id in img_ids:
                rle_list = train_tf.loc[train_tf['ImageId'] == id, 'EncodedPixels'].tolist()
                labels.append(generate_mask(rle_list))
            yield np.array(features),np.array(labels)

def visualize_imgs_with_masks(imgs,masks):
    num_img = len(imgs)
    plt.figure(figsize=(100, 300))
    for i in range(num_img):
        plt.subplot(num_img+1, 1, i+1)
        plt.imshow(imgs[i])
        plt.imshow(masks[i],alpha=0.4)

def withship_noship_split(train_tf):
    train_tf_count = train_tf.groupby(["ImageId"]).count()
    train_tf_count.rename(columns={"EncodedPixels": "ship_count"}, inplace=True)

    ids_withship = train_tf_count.loc[train_tf_count["ship_count"] > 0].index.tolist()
    ids_noship = train_tf_count.loc[train_tf_count["ship_count"] == 0].index.tolist()
    return ids_withship, ids_noship

def train_val_split(ids_withship,ids_noship):
    train_ids_withship,val_ids_withship = train_test_split(ids_withship,test_size=0.05,random_state=1234)
    train_ids_noship,val_ids_noship = train_test_split(ids_noship,test_size=0.02,random_state=1234)
    return train_ids_withship+train_ids_noship, val_ids_withship+val_ids_noship


# print(len(ids_withship))
# print(len(ids_noship))
train_tf = pd.read_csv(config.TRAIN_FILE)
ids_withship,ids_noship = withship_noship_split(train_tf)
train_val_split(ids_withship,ids_noship)
