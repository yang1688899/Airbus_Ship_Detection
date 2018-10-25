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

#data
img_paths = glob.glob(config.TRAIN_DIR+"*.jpg")
train_paths,val_paths = train_test_split(img_paths,test_size=0.1,random_state=1234)
train_tf = pd.read_csv(config.TRAIN_FILE)

train_gen = utils.data_generator(train_paths,train_tf,batch_size=config.BATCH_SIZE)
val_gen = utils.data_generator(val_paths,train_tf,batch_size=config.BATCH_SIZE,is_shuffle=False)

#build model
save_model_path = "./keras.model"
input_layer = Input(config.INPUT_SHAPE)
output_layer = network.network(input_layer, 16,0.5)

model = Model(input_layer, output_layer)
model.summary()

opt = optimizers.adam(lr=0.001)
model.compile(loss="binary_crossentropy",optimizer=opt)

early_stopping = EarlyStopping(patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1)
epochs = 200

history = model.fit_generator(train_gen,
                    steps_per_epoch=ceil(len(train_paths)/ config.BATCH_SIZE),
                    validation_data=val_gen,
                    validation_steps=ceil(len(val_paths)/config.BATCH_SIZE),
                    epochs=epochs,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                    verbose=1)

utils.save_to_pickle(history,"history.p")


