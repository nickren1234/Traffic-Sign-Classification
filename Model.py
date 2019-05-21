from Load import load_data
import os
import tensorflow as tf
from tensorflow.python.keras import layers, utils
import numpy as np
from skimage.color import rgb2gray

# where that images are stored
ROOT_PATH = "/BTS/"
train_data_directory = os.path.join(ROOT_PATH, "signs/Training")
test_data_directory = os.path.join(ROOT_PATH, "Signs/Testing")

images, labels = load_data(train_data_directory)

images = np.array(images)
labels = np.array(labels)

# shuffle the images
new_order = np.arange(images.shape[0])
np.random.shuffle(new_order)
images = images[new_order]
labels = labels[new_order]

num_classes = len(np.unique(labels))
size = len(images)

# split testing and training sets
(X_train, X_test) = images[(int)(0.1*size):], images[:(int)(0.1*size)]

(Y_train, Y_test) = labels[(int)(0.1*size):], labels[:(int)(0.1*size)]

Y_train = utils.to_categorical(Y_train, num_classes)
Y_test = utils.to_categorical(Y_test, num_classes)


model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=(28, 28, 3)))
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (2, 2),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (2, 2),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(62, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=25, epochs=200, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

# Save weights
model.save_weights("/BTS/weights/first_run.h5")
