#supress warnings
import warnings
warnings.filterwarnings('ignore')

#load keras and  numpy
import keras
import numpy as np


# load minst data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# load matplotlib for visualization
import matplotlib.pyplot as plt


#prepare data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)


#plot first image in training data
plt.imshow(X_train[0][:,:,0], cmap='gray')
plt.show()


#build model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#train model
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)


#evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#perfrom prediction
prediction = model.predict(X_test)
print(prediction)
print(np.argmax(prediction[0]))
print(np.argmax(Y_test[0]))
print(np.argmax(prediction[1]))
print(np.argmax(Y_test[1]))
print(np.argmax(prediction[2]))
print(np.argmax(Y_test[2]))
print(np.argmax(prediction[3]))
print(np.argmax(Y_test[3]))


#plot first image in test data
plt.imshow(X_test[0][:,:,0], cmap='gray')
plt.show()


#plot first image in test data with prediction
plt.imshow(X_test[0][:,:,0], cmap='gray')
plt.show()