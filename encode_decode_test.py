import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import config
import utils

id = "00021ddc3.jpg"

train_tf = pd.read_csv(config.TRAIN_FILE)

rle_list = train_tf.loc[train_tf['ImageId'] == id, 'EncodedPixels'].tolist()
decode_mask = utils.generate_mask(rle_list,shape=config.MASK_SHAPE)

rle_list_re = utils.multi_rle_encode(decode_mask.reshape([768,768,1]))
decode_mask_re = utils.generate_mask(rle_list_re,shape=config.MASK_SHAPE)

print(len(rle_list))
print(len(rle_list_re))

plt.figure(figsize=(20, 40))
plt.subplot(2, 2, 1)
plt.imshow(decode_mask)
plt.subplot(2, 2, 2)
plt.imshow(decode_mask_re)