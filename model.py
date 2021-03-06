import csv
print("Loading cv2...")
import cv2
import numpy as np
print("Loading tensorflow...")
import tensorflow as tf
print("Loading numpy...")
import numpy as np

print("Loading keras...")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPool2D, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import random
exit(0)

DATA_PATH = "/home/mquinteiro/proyectos/"

#the image loading and augment augmentation
def loadImages(path,correction=0.2):
    print("Loading images...")

    lines = []
    images = []
    measurements = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        #for each line we will use both side cameras and
        #drop some images to mormalize the distribucions of steering well angles.
        for line in reader:
            lines.append(line)

            if abs(float(line[3]))<=0.04:
                #all of 6 but one "almous strigth images
                #are droped.
                if random.randint(0,6)==0:
                    #for the one that is not droped we save the
                    #angle
                    measurements.append(float(line[3]))
                    #and the inversed angle
                    measurements.append(-float(line[3]))
                    #as well as image
                    image = cv2.imread(line[0])
                    images.append(image)
                    #and flipped image.
                    images.append(np.fliplr(image))

            #the value of correction define how aggresive the movemet will be.
            #adding images from the side cams,
            #that camaras could help the model to
            #return to the center of the scena.

            ##start with normal images, first left image with + correction
            measurements.append(float(line[3])+correction)
            measurements.append(float(line[3])-correction)

            # coninue with mirrored images old left first
            measurements.append(-(float(line[3])+correction))
            measurements.append(-(float(line[3])-correction))

            #load images with openCV
            image2 = cv2.imread(line[1])
            image3 = cv2.imread(line[2])
            #apend normal images
            images.append(image2)
            images.append(image3)
            # apend flipped images
            images.append(np.fliplr(image2))
            images.append(np.fliplr(image3))
    return np.array(images), np.array(measurements)

#this function is a wrapper to select witch architecture we will use.
def defineDriver(model='MQ'):
    if model=='MQ' :
        modelF = mqModel
    elif model=='NVidia':
        modelF = nVidiaModel
    else:
        print("Model "+ model + " not found")
        exit(-1)

    return modelF()


#My own architecture, at the end NVidia works better... surprising :)
def mqModel():

    print("Creating the model...")

    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,35), (0,0))))
    model.add(Conv2D(3, (1, 1), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(Conv2D(6,(5,5),activation='relu'))
    model.add(MaxPool2D())
    #model.add(MaxPool2D())
    model.add(Conv2D(6,(3,3),activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))
    model.add(Flatten())
#    model.add(Dropout(0.3))
#    model.add(Dense(128))
#    model.add(Dropout(0.3))
    #model.add(Dense(84))
    #model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer=Adam(lr=0.001))

    return model

def nVidiaModel():
    print("Creating NVidea model...")
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 35), (0, 0))))
    model.add(BatchNormalization())
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    #model.compile(loss='mse', optimizer="adam")
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

def trainDriver(model, X_train,y_train, epochs):
    print("Training the model...")
    model.fit(X_train,y_train,validation_split=0.20, shuffle=True,epochs=epochs)


import matplotlib.pyplot as plt
def main():
    #parameters for histograms
    plt.ion()
    bin_size = 0.1;
    min_edge = -0.95
    max_edge = 0.95
    N = 21

    #selected architecture.
    NNType = 'NVidia'
    #NNType = 'MQ'

    # create de model
    model = defineDriver(NNType)
    X_train, y_train = loadImages(DATA_PATH + "2Vueltas/", 5 / 25.0)
    trainDriver(model, X_train,y_train,5)
    model.save('model.h5')

    # old comvinations of different datasets.

    #X_train, y_train = loadImages(DATA_PATH + "C2TN1/", 2 / 25.0)
    #X_train = np.concatenate([X_train, X_train2], axis=0)
    #y_train = np.concatenate([y_train, y_train2], axis=0)
    #totalLabels = np.concatenate([totalLabels, y_train], axis=0)
    #plt.hist(y_train, bin_list)
    '''plt.draw()
    trainDriver(model, X_train, y_train, 8)
    model.save('model.h5')
    exit(0)
    X_train, y_train = loadImages(DATA_PATH + "PC1/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels,y_train],axis=0)
    plt.hist(y_train, bin_list)
    trainDriver(model, X_train, y_train, 2)
    model.save('model.h5')
    X_train, y_train = loadImages(DATA_PATH+"2N/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    trainDriver(model, X_train, y_train, 2)
    X_train, y_train = loadImages(DATA_PATH + "linux_sim/linux_sim/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    plt.hist(y_train, bin_list)
    plt.draw()
    trainDriver(model, X_train, y_train, 2)
    model.save('model.h5')
    X_train, y_train = loadImages(DATA_PATH + "CURVES/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    plt.hist(y_train, bin_list)
    plt.draw()
    trainDriver(model, X_train, y_train, 5)
    model.save('model.h5')
    '''

    #show tipical complicated situations.

    testImages = ["center_2017_07_09_16_00_41_503.jpg", "center_2017_07_09_16_00_47_672.jpg","center_2017_07_09_16_00_49_866.jpg",
                  "right_2017_07_09_16_00_51_117.jpg"]
    for imageName in testImages:
        image = cv2.imread("../IMG/" +imageName)
        prediction = model.predict(image[None, :, :, :])
        print(prediction)



if __name__ == '__main__':
    main()


