import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import scipy
import PIL
from PIL import Image, ImageFilter
from numpy import random
import skimage
batch_size = 256

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# resize to 32x32
temp = []
for x in x_train:
	temp .append(np.array(Image.fromarray(x).resize((32, 32))))

x_train = np.asarray(temp)

# resize to 32x32, and also add rotated, gaussian noise to x_test
temp = []
rotations = [-40,-30,-20,-10, 0, 10, 20, 30, 40]
noise = [0.01, 0.1, 1]

x_test_rotated = [[] for x in range(0,len(rotations))]
x_test_noise = [[] for x in range(0,len(noise))]

for i, x in enumerate(x_test):
	x = np.array(Image.fromarray(x).resize((32, 32)))
	temp.append(x)

	for j, rotation in enumerate(rotations):
		x_test_rotated[j].append(np.array(Image.fromarray(x).rotate(rotation)))

for i, x in enumerate(x_test):
	for j, std in enumerate(noise):
		x = skimage.util.random_noise(image = x, mode= 'gaussian', clip=True, mean = 0.0, var = std)
		x = np.array(Image.fromarray(x).resize((32, 32)))
		'''
		x = np.array(Image.fromarray(x).resize((32, 32)))
		x = x + np.random.gauss( 0.0, std, x.shape)
		x = np.clip(x , 0, 255)
		'''
		x_test_noise[j].append(x)

x_test = np.asarray(temp)

# reshape to 32 * 32 * 1

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

for i in range(0, len(x_test_rotated)):
	x_test_rotated[i] = np.asarray(x_test_rotated[i])
	x_test_rotated[i] = x_test_rotated[i].reshape(x_test_rotated[i].shape[0], 32, 32, 1)
	
for i in range(0, len(x_test_noise)):
	x_test_noise[i] = np.asarray(x_test_noise[i])
	x_test_noise[i] = x_test_noise[i].reshape(x_test_noise[i].shape[0], 32, 32, 1)

input_shape = (32, 32, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# number of classes is 10
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



model = Sequential()

# Create model
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape = (32, 32, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dense(4096, activation='relu'))

model.add(Dense(1000, activation='relu'))

model.add(Dense(10, activation='softmax'))


print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=1, validation_data=(x_test, y_test))


for i, rot in enumerate(rotations):
	score = model.evaluate(x_test_rotated[i], y_test, verbose=0)
	print('Test loss, rotated ', rot, ' loss:',  score[0])
	print('Test accuracy, rotated ', rot, ' acc:', score[1])
  
for i, std  in enumerate(noise):
	score = model.evaluate(x_test_noise[i], y_test, verbose=0)
	print('Test loss, std  ', std , ' loss:',  score[0])
	print('Test accuracy, std  ', std , ' acc:', score[1])
	
'''
Output: 
Epoch 1/5
60000/60000 [==============================] - 745s 12ms/step - loss: 1.5332 - acc: 0.4981 - val_loss: 0.7584 - val_acc: 0.7484
Epoch 2/5
60000/60000 [==============================] - 739s 12ms/step - loss: 0.5508 - acc: 0.8156 - val_loss: 0.4165 - val_acc: 0.8682
Epoch 3/5
60000/60000 [==============================] - 739s 12ms/step - loss: 0.3331 - acc: 0.8961 - val_loss: 0.2449 - val_acc: 0.9241
Epoch 4/5
60000/60000 [==============================] - 740s 12ms/step - loss: 0.2313 - acc: 0.9297 - val_loss: 0.1863 - val_acc: 0.9419
Epoch 5/5
60000/60000 [==============================] - 740s 12ms/step - loss: 0.1787 - acc: 0.9451 - val_loss: 0.1442 - val_acc: 0.9561
Test loss, rotated  -40  loss: 8.65370829925537
Test accuracy, rotated  -40  acc: 0.4599
Test loss, rotated  -30  loss: 5.19091490688324
Test accuracy, rotated  -30  acc: 0.6739
Test loss, rotated  -20  loss: 2.335776023155451
Test accuracy, rotated  -20  acc: 0.8534
Test loss, rotated  -10  loss: 1.0328960903696607
Test accuracy, rotated  -10  acc: 0.935
Test loss, rotated  0  loss: 0.7679140400262552
Test accuracy, rotated  0  acc: 0.9514
Test loss, rotated  10  loss: 1.0359044186836108
Test accuracy, rotated  10  acc: 0.9348
Test loss, rotated  20  loss: 2.313044224023819
Test accuracy, rotated  20  acc: 0.854
Test loss, rotated  30  loss: 4.991883773231506
Test accuracy, rotated  30  acc: 0.6871
Test loss, rotated  40  loss: 8.535515224456788
Test accuracy, rotated  40  acc: 0.4668
Test loss, std   0.01  loss: 0.16498079838752747
Test accuracy, std   0.01  acc: 0.9505
Test loss, std   0.1  loss: 1.183865225124359
Test accuracy, std   0.1  acc: 0.6144
Test loss, std   1  loss: 7.666171592712402
Test accuracy, std   1  acc: 0.1026
'''
