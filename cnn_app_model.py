import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from PIL import UnidentifiedImageError
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#Hyperparameters
optimizer = ['Adam'] 
epochs = 10 
stride = 1 
conv_layer = 3 
dense_layer = 1
dropout = 0.2 
layer_size = 32 


val_split = 0.3 

input_height = 150
input_width = 150

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30, 
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = val_split)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(directory = train_parent_directory,
                                                            target_size = (input_height,input_width),
                                                            batch_size=64,
                                                            class_mode = 'categorical',
                                                            subset='training')
    
    val_generator = train_datagen.flow_from_directory(directory = train_parent_directory,
                                                      target_size = (input_height,input_width),
                                                      batch_size=64,
                                                      class_mode = 'categorical',
                                                      subset='validation')
    
    test_generator = test_datagen.flow_from_directory(directory = test_parent_directory,
                                                      target_size = (input_height,input_width),
                                                      batch_size=5,
                                                      class_mode='categorical')
    
    return train_generator, val_generator, test_generator

train_dir = '<change directory>/datasets/train'
valid_dir = '<change directory>/datasets/valid'
test_dir = '<change directory>/datasets/test'
train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

early_stop_accuracy = EarlyStopping(monitor="accuracy",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True
)

early_stop_loss = EarlyStopping(monitor="loss",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True
)

early_stop_val_loss = EarlyStopping(monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True
)

early_stop_val_accuracy = EarlyStopping(monitor="val_accuracy",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True
)

#Code to identify corrupted images 
path = Path("<change directory>/datasets/test").rglob("*.jpg") 
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)

NAME = "{}-new-{}-nodes-{}-dense-{}-dropout".format(conv_layer, layer_size, dense_layer, dropout)

tb_callback = tf.keras.callbacks.TensorBoard(
log_dir= "logs/%s" % NAME ,
histogram_freq=0,
write_graph=True,
write_images=False,
write_steps_per_second=False,
update_freq="epoch",
profile_batch=0,
embeddings_freq=0,
embeddings_metadata=None,
)
   
    
model = Sequential([
    #Input Layer
    Conv2D(filters=layer_size, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(input_height,input_width,3)),
    MaxPool2D(pool_size=(2, 2), strides=stride),
    
    #Hidden Layers
    Conv2D(filters=layer_size, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=stride),

    Conv2D(filters=layer_size, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=stride),

    Conv2D(filters=layer_size, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=stride),
    
    #Dropout
    Dropout(dropout),
    
    #Flatten
    Flatten(),
    
    #Dense Layer 
    Dense(units=layer_size, activation='relu'),

    #Output Layer
    Dense(units=3, activation='softmax')
])

model.summary()
print(f'Building Model: {NAME}')
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=epochs,
    callbacks=[early_stop_accuracy, early_stop_loss, early_stop_val_loss, early_stop_val_accuracy, tb_callback],
    verbose=1
    )
tf.keras.models.save_model(model,"<change directory>/models/cnn_model_%s.hdf5" % NAME)
tf.keras.backend.clear_session()


