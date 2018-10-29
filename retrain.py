from keras.models import load_model
from keras import Input,optimizers,Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from math import ceil

import utils
import config
import network
from loss_function import focal_dice_loss,my_iou_metric


#save_file
save_model_path = "./save/unet_384_facal_dice.model"

#data
train_tf = pd.read_csv(config.TRAIN_FILE)

save_obj = utils.load_pickle("./split_ids.p")
train_ids,val_ids = save_obj

train_paths = [config.TRAIN_DIR+id for id in train_ids]
val_paths = [config.TRAIN_DIR+id for id in val_ids]

train_gen = utils.data_generator(train_paths,train_tf,batch_size=config.BATCH_SIZE)
val_gen = utils.data_generator(val_paths,train_tf,batch_size=config.BATCH_SIZE,is_shuffle=False)

#load model
model = load_model(save_model_path,custom_objects={'focal_dice_loss': focal_dice_loss,
                                                        'my_iou_metric': my_iou_metric})
model.summary()
print("train samples: %s"%len(train_ids))
print("valid samples: %s"%len(val_ids))

opt = optimizers.adam(lr=0.00025)
model.compile(loss=focal_dice_loss,optimizer=opt,metrics=[my_iou_metric])


early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=3, verbose=1)

model_checkpoint = ModelCheckpoint(save_model_path,monitor='val_my_iou_metric',
                                   mode = 'max',save_best_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=2, min_lr=0.00001, verbose=1)
epochs = 50

history = model.fit_generator(train_gen,
                    # steps_per_epoch=ceil(len(train_paths)/ config.BATCH_SIZE),
                    steps_per_epoch=10000,
                    validation_data=val_gen,
                    validation_steps=ceil(len(val_paths)/config.BATCH_SIZE),
                    epochs=epochs,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                    initial_epoch=15,
                    verbose=2)

utils.save_to_pickle(history,"history.p")