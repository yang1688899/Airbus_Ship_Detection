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
from loss_function import focal_dice_loss,dice,IoU

#save_file
save_model_path = "./save/unet_384_facal_dice.model"

#data
train_tf = pd.read_csv(config.TRAIN_FILE)
ids_withship,ids_noship = utils.withship_noship_split(train_tf)
train_ids,val_ids = utils.train_val_split(ids_withship,ids_noship)
train_paths = [config.TRAIN_DIR+id for id in train_ids]
val_paths = [config.TRAIN_DIR+id for id in val_ids]

train_gen = utils.data_generator(train_paths,train_tf,batch_size=config.BATCH_SIZE)
val_gen = utils.data_generator(val_paths,train_tf,batch_size=config.BATCH_SIZE,is_shuffle=False)

batch = next(train_gen)
print(train_gen.shape)

#build model
input_layer = Input(config.INPUT_SHAPE)
output_layer = network.network(input_layer, 16,0.5)

model = Model(input_layer, output_layer)
model.summary()

opt = optimizers.adam(lr=0.001)
model.compile(loss=focal_dice_loss,optimizer=opt,metrics=[dice])

early_stopping = EarlyStopping(monitor='val_dice', mode = 'max',patience=3, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_path,monitor='val_dice',
                                   mode = 'max',save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_dice', mode = 'max',factor=0.5, patience=2, min_lr=0.00001, verbose=1)
epochs = 20

history = model.fit_generator(train_gen,
                    steps_per_epoch=ceil(len(train_paths)/ config.BATCH_SIZE),
                    validation_data=val_gen,
                    validation_steps=ceil(len(val_paths)/config.BATCH_SIZE),
                    epochs=epochs,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                    verbose=1)

utils.save_to_pickle(history,"history.p")


