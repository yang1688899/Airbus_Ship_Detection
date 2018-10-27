from keras.models import load_model
import pandas as pd
import numpy as np

import utils
import config
from loss_function import focal_dice_loss,mean_iou

id = "00021ddc3.jpg"

train_tf = pd.read_csv(config.TRAIN_FILE)

rle_list = train_tf.loc[train_tf['ImageId'] == id, 'EncodedPixels'].tolist()
mask = utils.generate_mask(rle_list,shape=config.MASK_SHAPE)

save_model_path = "./save/unet_384_facal_dice.model"
model = load_model(save_model_path,custom_objects={'focal_dice_loss': focal_dice_loss,
                                                        'mean_iou': mean_iou})
img = utils.load_img(config.TRAIN_DIR+id)
feature = np.array([img])
predict = model.predict(feature)

print(predict.shape)