import glob
from keras.models import load_model
import numpy as np
import cv2
import pandas as pd

import config
import utils
from loss_function import focal_dice_loss,my_iou_metric

#load model
save_model_path = "./save/unet_384_facal_dice.model"
model = load_model(save_model_path,custom_objects={'focal_dice_loss': focal_dice_loss,
                                                        'my_iou_metric': my_iou_metric})

test_paths = glob.glob(config.TEST_DIR+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]

summit_data = []
for offset in range(0,len(test_paths),config.BATCH_SIZE*2):
    batch_paths = test_paths[offset:offset+config.BATCH_SIZE*2]
    batch_ids = test_ids[offset:offset+config.BATCH_SIZE*2]
    test_features = np.array([utils.load_img(path) for path in batch_paths])
    test_features_reflect = np.array([np.fliplr(x) for x in test_features])

    preds_test = model.predict(test_features).reshape(-1, config.TARGET_SIZE, config.TARGET_SIZE)
    preds_test2_refect = model.predict(test_features_reflect).reshape(-1, config.TARGET_SIZE, config.TARGET_SIZE)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
    preds_test = preds_test/2.

    collect = []
    for i,id in enumerate(batch_ids):
       rle_list =  utils.multi_rle_encode( utils.upsample(np.round(preds_test[i]>0.5)) )

       if rle_list:
           for rle in rle_list:
               collect.append([id,rle])
       else:
           collect.append([id, ""])
    summit_data.extend(collect)

sub = pd.DataFrame(summit_data)
sub.columns = ['ImageId','EncodedPixels']
sub.to_csv("./submit.csv",index=False)



