import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import pandas as pd
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# preprocessing

data = pd.read_csv('training_data3/driving_log.csv',
names = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed'])

# Remove speed, brake and time
data = data.drop(['throttle', 'brake', 'speed'], axis = 1)


# Check how our data is distributed
#histogram = data_c.hist(column = 'angle', bins = 12)

# Remove data so it contains equal amounts of left, right, forward
# Remove 80% of forward data
indx = data['angle'] == 0
zero_angles = data[indx]
zero_angles = zero_angles.sample(frac = 0.1).reset_index(drop=True)
data = data[np.invert(indx)].reset_index(drop=True)

data = pd.concat([zero_angles, data], ignore_index = True)
#histogram2 = data_c.hist(column = 'angle', bins = 12)

# Divide dataset into left, right and center
data_c = data.drop(['left', 'right'], axis = 1)

#  May want to remove some of the data here as well
data_l = data.drop(['center', 'right'], axis = 1)
data_r = data.drop(['center', 'left'], axis = 1)


# Correct offset for left and right datasets
offset = 0.1 # Tunable parameter
data_r['angle'] = data_r['angle'] - offset
data_l['angle'] = data_l['angle'] + offset

# Put dataset together as one set of images
data_r = data_r.rename({'right': 'image'}, axis = 'columns')
data_l = data_l.rename({'left': 'image'}, axis = 'columns')
data_c = data_c.rename({'center': 'image'}, axis = 'columns')

data = pd.concat([data_c, data_l, data_r], ignore_index = True)
histogram3 = data.hist(column='angle', bins = 12)
plt.show()

# Read images to arrays
def readImg(image_path):
    # Using matplotlib
    img  = cv2.imread(image_path, 1)

    # remove bottom 20 and top 20 pixels
    # These contain background and part of the car
    img = img[20:-20,:,:]

    #resize to fit dave-2 network architecture
    img = cv2.resize(img, (200,66), interpolation = cv2.INTER_LINEAR)
    return img

xdata = []
for img in data["image"]:
    xdata.append(readImg(img))
xdata = np.asarray(xdata)
ydata = np.asarray(data["angle"])


# Divide into training and validation set
indx = np.arange(0, xdata.shape[0])
np.random.shuffle(indx)

#Validation split should be small (5%)
train_length = int(xdata.shape[0] * 0.95)
xtrain = xdata[indx[0:train_length]]
ytrain = ydata[indx[0:train_length]]

xval = xdata[indx[train_length:]]
yval = ydata[indx[train_length:]]

print(xtrain.shape)

# Preprocess images
# TODO: image brightness changer


# Network
model = Sequential()
# Normalizes input
model.add(layers.BatchNormalization(input_shape=(66, 200,  3)))

# Convolutional layers with 5x5 kernel and 2 stride
model.add(layers.Conv2D(24, (5,5), strides = (2,2), activation='relu'))
model.add(layers.Conv2D(36, (5,5), strides =(2,2), activation='relu'))
model.add(layers.Conv2D(48, (5,5), strides =(2,2), activation='relu'))

# Non strided convolution with 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), strides =(1, 1), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), strides =(1, 1), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dense(50))
model.add(layers.Dense(10))
model.add(layers.Dense(1, activation='tanh', name='output'))


adam = optimizers.Adam(lr = 0.001, decay = 0.0)
model.compile(optimizer = adam, loss = losses.mean_squared_error)

model.summary()



# Train the network
BATCH_SIZE = 100
EPOCHS = 10
SAMPLES = len(xtrain)
datagen = ImageDataGenerator(shear_range = 0.1)
model.fit_generator(datagen.flow(xtrain, ytrain, batch_size = BATCH_SIZE),
samples_per_epoch = SAMPLES, epochs = EPOCHS, validation_data = (xval, yval))



model.save("model.h5")
