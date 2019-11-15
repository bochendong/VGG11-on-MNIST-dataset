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
from keras import regularizers
import random
batch_size = 256

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# resize to 32x32

rotations = [-40,-30,-20,-10, 0, 10, 20, 30, 40]
noise = [0.01, 0.1, 1]
random.seed(69)

temp = []
for x in x_train:
	if random.random() >= 0.5:
		rand_rot = rotations[random.randint(0, 8)]
		x = np.array(Image.fromarray(x).rotate(rand_rot))

	if random.random() >= 0.5:
		rand_var = noise [random.randint(0, 2)]
		x = skimage.util.random_noise(image = x, mode= 'gaussian', clip=True, mean = 0.0, var = rand_var)
	
	temp.append(np.array(Image.fromarray(x).resize((32, 32))))

x_train = np.asarray(temp)



# resize to 32x32, and also add rotated, gaussian noise to x_test
temp = []
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


# Create model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_regularizer=regularizers.l2(0.0005)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0005)))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Maxpool, Fully Connected Layers & Softmax
model.add(Flatten())
model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))

model.add(Dense(10, kernel_regularizer=regularizers.l2(0.0005), activation='softmax'))


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
60000/60000 [==============================] - 766s 13ms/step - loss: 6.3248 - acc: 0.1762 - val_loss: 5.7244 - val_acc: 0.3707
Epoch 2/5
60000/60000 [==============================] - 847s 14ms/step - loss: 5.0810 - acc: 0.3510 - val_loss: 3.8651 - val_acc: 0.7488
Epoch 3/5
60000/60000 [==============================] - 788s 13ms/step - loss: 4.4732 - acc: 0.4251 - val_loss: 3.3298 - val_acc: 0.8164
Epoch 4/5
60000/60000 [==============================] - 1515s 25ms/step - loss: 4.0870 - acc: 0.4571 - val_loss: 2.8896 - val_acc: 0.8800
Epoch 5/5
60000/60000 [==============================] - 1726s 29ms/step - loss: 3.7806 - acc: 0.4900 - val_loss: 2.6328 - val_acc: 0.8947
Test loss, rotated  -40  loss: 7.590001333618164
Test accuracy, rotated  -40  acc: 0.6683
Test loss, rotated  -30  loss: 6.107901028442383
Test accuracy, rotated  -30  acc: 0.76
Test loss, rotated  -20  loss: 4.893476404571533
Test accuracy, rotated  -20  acc: 0.8366
Test loss, rotated  -10  loss: 4.260469832611084
Test accuracy, rotated  -10  acc: 0.8755
Test loss, rotated  0  loss: 3.979454473114014
Test accuracy, rotated  0  acc: 0.8936
Test loss, rotated  10  loss: 4.161555628967285
Test accuracy, rotated  10  acc: 0.8814
Test loss, rotated  20  loss: 4.685032397460938
Test accuracy, rotated  20  acc: 0.8493
Test loss, rotated  30  loss: 5.587607210540772
Test accuracy, rotated  30  acc: 0.7922
Test loss, rotated  40  loss: 6.8248119606018065
Test accuracy, rotated  40  acc: 0.714
Test loss, std   0.01  loss: 2.6727857151031493
Test accuracy, std   0.01  acc: 0.8872
Test loss, std   0.1  loss: 4.026104756164551
Test accuracy, std   0.1  acc: 0.4644
Test loss, std   1  loss: 9.704185195922852
Test accuracy, std   1  acc: 0.1599
'''