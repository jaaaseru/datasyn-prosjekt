import numpy as np
# import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# preprocessing

data = pd.read_csv('training_data/driving_log.csv',
names = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed'])

# Remove speed, brake and time
data = data.drop(['throttle', 'brake', 'speed'], axis = 1)
# Divide dataset into left, right and center
data_c = data.drop(['left', 'right'], axis = 1)

#  May want to remove some of the data here as well
data_l = data.drop(['center', 'right'], axis = 1)
data_r = data.drop(['center', 'left'], axis = 1)

# Check how our data is distributed
histogram = data_c.hist(column = 'angle', bins = 12)
#plt.show()

# Remove data so it contains equal amounts of left, right, forward
# Remove 80% of forward data
indx = data_c['angle'] == 0
zero_angles = data_c[indx]
zero_angles = zero_angles.sample(frac = 0.2).reset_index(drop=True)
data_c = data_c[np.invert(indx)].reset_index(drop=True)

data_c = pd.concat([zero_angles, data_c], ignore_index = True)
histogram2 = data_c.hist(column = 'angle', bins = 12)
#plt.show()

# Correct offset for left and right datasets
offset = 0.1 # Tunable parameter
data_r['angle'] = data_r['angle'] - offset
data_l['angle'] = data_l['angle'] + offset

# Put dataset together as one set of images
data_r = data_r.rename({'right': 'image'}, axis = 'columns')
data_l = data_l.rename({'left': 'image'}, axis = 'columns')
data_c = data_c.rename({'center': 'image'}, axis = 'columns')

data = pd.concat([data_c, data_l, data_r], ignore_index = True)
data = data.drop('index', axis = 1)
print(data.axes)
print(data)
histogram3 = data.hist(column='angle', bins = 12)
#plt.show()


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



model.save("model.h5")
