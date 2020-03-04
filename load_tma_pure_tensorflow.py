#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
import pathlib 
import pandas as pd


from tensorflow import keras

# Abgewandeltes tutorial "Load images" using tf.data von: https://www.tensorflow.org/tutorials/load_data/images
# hier komme ich nicht weiter, da ich das befüllte dataset nicht an ein neuronales Netz weitergeben kann 
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    file_name = parts[-1]
    label = tf.strings.split(file_name, '_')
    #return label[0] == CLASS_NAMES
    return tf.strings.to_number(label[0], out_type=tf.int32)

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


print(tf.__version__)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


data_dir = pathlib.Path('./color_images/')

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

for f in list_ds.take(3):
     print(f.numpy())

# %% 

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(3):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())



# %%

#train_dataset = labeled_ds.take(780)
#val_dataset = labeled_ds.skip(780)

#train_ds = prepare_for_training(labeled_ds)
#image_batch, label_batch = next(iter(train_ds))



# %%

train_ds = labeled_ds.shuffle(5000).batch(32)

# %%

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3)
])

# %%

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
# das funktioniert nicht, keine Ahnung warum nicht
model.fit(train_ds, epochs=10, validation_split = 0.3)

# %%
