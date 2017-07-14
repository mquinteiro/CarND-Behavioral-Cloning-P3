import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
import math as m

def plotFeatures(features):
    axs = np.empty(features.shape)
    f = np.empty(features.shape)
    for feature in range(0,features.shape[0]):
        f, axs = plt.subplots(m.ceil(features[feature].shape[2]/4), 4)
        for subIDX in range(0,features.shape[3]):
            if features.shape[3]>4:
                axs[subIDX//4,subIDX%4].imshow(features[feature,:,:,subIDX],cmap="gray")
            else:
                axs[subIDX].imshow(features[feature, :, :,subIDX], cmap="gray")

model = load_model("model.h5")
img = cv2.cvtColor(cv2.imread("center_2017_07_09_16_00_51_117.jpg"),  cv2.COLOR_BGR2RGB)
plt.imshow(img)

for layer in model.layers:
    print(layer.name)
    model_intermedean= Model(input=model.input, output=layer.output)
    features = model_intermedean.predict(img[None, :, :, :])
    plotFeatures(features)

