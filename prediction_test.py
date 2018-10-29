from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import utils
import config
from loss_function import focal_dice_loss,my_iou_metric

id = "00113a75c.jpg"

train_tf = pd.read_csv(config.TRAIN_FILE)

rle_list = train_tf.loc[train_tf['ImageId'] == id, 'EncodedPixels'].tolist()
label_mask = utils.generate_mask(rle_list,shape=config.MASK_SHAPE)

save_model_path = "./save/unet_384_facal_dice.model"
model = load_model(save_model_path,custom_objects={'focal_dice_loss': focal_dice_loss,
                                                        'my_iou_metric': my_iou_metric})
img = utils.load_img(config.TRAIN_DIR+id)
feature = np.array([img])
predict = model.predict(feature)
pred_mask = np.array(predict>0.5,dtype=np.int32)[0]

print(label_mask.shape)
print(pred_mask.shape)

print(pred_mask)

pred_mask_re = utils.upsample(pred_mask)

print(pred_mask_re)

plt.figure(figsize=(20, 40))
plt.subplot(2, 2, 1)
plt.imshow(pred_mask_re.reshape(config.ORIGIN_SIZE,config.ORIGIN_SIZE))
plt.subplot(2, 2, 2)
plt.imshow(pred_mask.reshape(config.TARGET_SIZE,config.TARGET_SIZE))