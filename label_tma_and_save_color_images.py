#%%

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
import shutil
import math
import uuid

# Extrahiert für einen Dateinamen aus den scores (csv Datei) die Bewertung der Pathologen
# Der Dateiname muss strukturiert aufgebaut sein
# der letzte übergeordnete Ordner vor dem Datainamen muss die Färbung sein und mit dem entsprechenden Feld in scores korrespondieren
def get_label_and_filename(image_path, scores, anonymous = True):

    
    a = os.path.basename(os.path.dirname(image_path))

    staining = a.split("_")[0]
   
    TMA = (a.split("_")[-1])
    # parse filename A-1.jpg -> a and 1 definiert die Position der Stanze auf dem TMA
    a = os.path.splitext(os.path.basename(image_path))[0]
    pos_char = (a.split("-")[0]).lower()
    pos_num = (a.split("-")[1])
    


    label = (scores[(scores['TMA'] == TMA) & (scores['pos_char'] == pos_char) & (scores['pos_num'] == pos_num)][staining])
    if(math.isnan(label) == True): 
        label = 99.0
    if(anonymous == True):
        unique_filename = str(uuid.uuid4())
        new_filename = str(int(label)) + "_" + unique_filename
    else:
        # hier kann später noch das Pseudonym eingebaut werden
        new_filename = str(int(label)) + "_" + staining + "_" + TMA + "_" + pos_char + "-" + pos_num  
    return int(label), new_filename

# csv-Datei mit den Bewertungen der PAthologen laden    
sc = pd.read_csv(os.path.join('..', 'prepare TMA images', 'Auswertung PANCALYZE TMAs NEU.csv'), sep=';', header=0, converters={'pos_num':str})

# umformatieren, da immer 2 Stanzen zum gleichen Patienten gehören und auf 2 Zeilen verteilt werden müssen
scores = \
(sc.set_index(sc.columns.drop('pos_num',1).tolist())
    .pos_num.str.split(',', expand=True)
    .stack()
    .reset_index()
    .rename(columns={0:'pos_num'})
    .loc[:, sc.columns]
)


input_dir = pathlib.Path('../prepare TMA images/kleine TMA Bilder/')
output_dir = pathlib.Path('./color_images/')


shutil.rmtree(output_dir)
os.makedirs(output_dir)

for dirname, dirs, files in os.walk(input_dir):
    for filename in files:
        label, new_filename = get_label_and_filename(os.path.join(dirname, filename), scores)
        # es erfolgt keine Extraktion der braunen DAB-Färbung
        #ihc_hed = rgb2hed(plt.imread(os.path.join(dirname, filename)))
        #img = ihc_hed[:, :, 2]
        old_name = os.path.join(dirname, filename)
 
        if not os.path.exists(os.path.join(output_dir, str(label))):
            os.makedirs(os.path.join(output_dir, str(label)))

        new_name = os.path.join(output_dir, str(label), new_filename + ".jpg")
        if(label < 99):
            shutil.copy(old_name, new_name)
 # %%
