#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import os
import getpass
from skimage import data
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
import pathlib 
import pandas as pd


from tensorflow import keras
import tensorflow_hub as hub

from tensorflow.keras import layers

# folgt dem tf for poets tutorial
# wie teilt man die Daten in ein Trainings- und Validierungsset?
# am liebsten würde ich Trainings- und Validierungsset speichern, damit man es für verschiedene Ansätze benutzen kann
# das Modell wird gespeichert
#  
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

IMAGE_SHAPE = (IMG_HEIGHT, IMG_WIDTH)


# %%

data_root = pathlib.Path('./color_images/')

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

# %%

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

# %%

# funktioniert im P-Netz nur mit
# set https_proxy=http://poppf:PWD@134.95.33.10:8080
# set http_proxy=http://poppf:PWD@134.95.33.10:8080

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# %%

feature_extractor_layer.trainable = True

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes)
])

model.summary()

# %%
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

# %%
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

# %%
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=20,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])


# funktioniert nicht
# model.save('my_model.h5') 

model.save('saved_model')
# %%
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
# Bild ist unter results abgespeichert
# %%
