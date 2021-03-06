# VGG11 on MNIST
Name: Bochen Dong

Email : dongbochen1218@icloud.com
## Detail:
Train a VGG11 net on the MNIST dataset.

Note that in this project, my input dimension is different from the VGG paper. So I resized each image in MNIST from its original size 28 × 28 to 32 × 32

ConvNet configurations:
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/VGG11.png"
        width="500" height="600">
	<p align="center">
</p>


### Resize to 32 * 32
```Python
temp = []
for x in x_train:
	temp.append(np.array(Image.fromarray(x).resize((32, 32))))

x_train = np.asarray(temp)

for i, x in enumerate(x_test):
	temp.append(np.array(Image.fromarray(x).resize((32, 32))))

x_test = np.asarray(temp)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
```

### Create model :
```Python
model = Sequential()

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

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.00001), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=1, validation_data=(x_test, y_test))
 
```
### Number of epochs vs Training and test loss, accuracy

Note that we use epochs = 5 here

|Epoch|loss|acc|val_loss|val_acc|
|---|---|---|---|---
|1|1.5332|0.4981|0.7584|0.7484|
|2|0.5508|0.8156|0.4165|0.8682|
|3|0.3331|0.8961|0.2449|0.9241|
|4|0.2313|0.9297|0.1863|0.9419|
|5|0.1787|0.9451|0.1442|0.9561|
#### Plot:
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/train_acc.png"
        width="400" height="300">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/train_loss.png"
        width="400" height="300">
<p align="center">
</p>
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/test_acc.png"
        width="400" height="300">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/test_loss.png"
        width="400" height="300">
<p align="center">
</p>

### Generate the rotated test set by rotating each image:

```Python
temp = []
for i, x in enumerate(x_test):
	x = np.array(Image.fromarray(x).resize((32, 32)))
	temp.append(x)

	for j, rotation in enumerate(rotations):
		x_test_rotated[j].append(np.array(Image.fromarray(x).rotate(rotation)))
```

### Test loss and accuracy vs the degree of rotation

|rotated|loss|acc|
|---|---|---
|-40|8.653|0.4599|
|-30|5.190|0.6739|
|-20|2.337|0.8534|
|-10|1.032|0.935|
|0|0.7679|0.9514|
|10|1.035|0.9348|
|20|2.313|0.854|
|30|4.991|0.6871|
|40|8.535|0.4668|
#### Plot:
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/r_test.png"
        width="450" height="300">
	<p align="center">
</p>

### Generate the Gaussian test set by adding Gaussian noise to each image:

```Python
for i, x in enumerate(x_test):
	for j, std in enumerate(noise):
		x = skimage.util.random_noise(image = x, mode= 'gaussian', clip=True, mean = 0.0, var = std)
		x = np.array(Image.fromarray(x).resize((32, 32)))

		x_test_noise[j].append(x)
 ```
 
### Test loss and accuracy vs Gaussian noise

|std|loss|acc|
|---|---|---
|0.01|0.164|0.9505
|0.1|1.183|0.6144
|1|7.666|0.1026

#### Plot:
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/v_test.png"
        width="450" height="300">
	<p align="center">
</p>

## Data_augmentation:
Have 50% probability to rotate the image with random degree in (40, -30, ... 30, 40) and 50% probability to add a random gaussian noice in (0.01, 0.1, 1). Then add L2 regularization with constant 0.005 to all conv layers (which is used in this [paper](https://arxiv.org/pdf/1409.1556.pdf))

### Add data augmentation to training set:
```Python
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
```
### Test loss and accuracy vs the degree of rotation
|rotated|loss|acc|
|---|---|---
|-40|7.590|0.6683|
|-30|6.107|0.76|
|-20|4.893|0.8366|
|-10|4.260|0.8755|
|0|3.979|0.8936|
|10|4.161|0.8814|
|20|4.685|0.8493|
|30|5.587|0.7922|
|40|6.824|0.714|


### Test loss and accuracy vs Gaussian noise
|std|loss|acc|
|---|---|---
|0.01|2.672|0.8872
|0.1| 4.026|0.4644
|1| 9.704|0.1599

### Add data augmentation vs did not add data augmentation:
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/r_test.png"
        width="400" height="300">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/r_acc2.png"
        width="400" height="300">
<p align="center">
</p>
<p align="center">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/v_test.png"
        width="400" height="300">
	<img src="https://github.com/bochendong/VGG11-on-MNIST-dataset/raw/master/image/v_acc2.png"
        width="400" height="300">
<p align="center">
</p>	

## For this project, the version used is:
|name|version|type|
|---|---|---
|absl-py |                  0.8.1 |                   py37_0  |
|astor   |                  0.8.0 |                   py37_0  |
|astroid  |                 2.3.2|                    py37_0  |
|blas    |                  1.0  |                  openblas  |
|c-ares      |              1.15.0  |          h1de35cc_1001  |
|ca-certificates  |         2019.10.16  |                  0  |
|certifi      |             2019.9.11  |              py37_0  |
|cloudpickle    |           1.2.2   |                   py_0   | 
|cycler       |             0.10.0   |                py37_0  |
|cytoolz       |            0.10.1  |         py37h0b31af3_0   |
|dask-core     |            2.7.0  |                    py_0    |
|decorator      |           4.4.1  |                    py_0  |  
|freetype        |          2.9.1  |              hb4e5f40_0  |
|gast      |                0.3.2   |                   py_0  |
|grpcio     |               1.16.1 |          py37h044775b_1  |
|h5py         |             2.9.0  |          py37h3134771_0 | 
|hdf5         |             1.10.4 |              hfa1e0ec_0  |
|imageio                   2.6.1  |                  py37_0  | 
|intel-openmp    |          2019.4  |                    233 | 
|isort        |             4.3.21    |               py37_0  |
|jpeg      |                9b       |            he5867d9_2  |
|keras       |              2.2.4     |                    0  |
|keras-applications|        1.0.8   |                   py_0  |
|keras-base      |          2.2.4   |                 py37_0  |
|keras-preprocessing |      1.1.0    |                  py_1  |
|kiwisolver   |             1.1.0    |        py37h0a44026_0  |
|lazy-object-proxy  |       1.4.3     |       py37h1de35cc_0  |
|libcxx   |                 4.0.1      |          hcfea43d_1  |
|libcxxabi  |               4.0.1       |        hcfea43d_1  |
|libedit  |                 3.1.20181209 |        hb402a30_0  |
|libffi   |                 3.2.1        |        h475c297_4  |
|libgfortran        |       3.0.1           |     h93005f0_2  |
|libopenblas        |       0.3.6            |    hdc02c5d_2  |
|libpng            |        1.6.37            |   ha441bb4_0  |
|libprotobuf      |         3.9.2              |  hd9629dc_0  |
|libtiff         |          4.1.0              |  hcb84e12_0  |
|markdown       |           3.1.1              |      py37_0  |
|matplotlib    |            3.1.1           | py37h54f8f79_0  |
|mccabe       |             0.6.1            |        py37_1  |
|mkl                     |  2019.4            |          233  |
|mkl-service            |   2.3.0            |py37hfbe908c_0  |
|mock                  |    3.0.5             |       py37_0  |
|ncurses              |     6.1               |  h0a44026_1  |
|networkx            |      2.4                |        py_0   | 
|nomkl              |       3.0                 |          0  |
|numpy             |        1.17.3           |py37hc29fe80_0  |
|numpy-base       |         1.17.3           |py37ha711998_0  |
|olefile         |          0.46              |       py37_0  |
|openssl        |           1.1.1d             |  h1de35cc_3  |
|pillow        |            6.2.1            |py37hb68e598_0  |
|pip          |             19.3.1            |       py37_0  |
|protobuf    |              3.9.2            |py37h0a44026_0  |
|pylint     |               2.4.3             |       py37_0  |
|pyparsing |                2.4.2              |        py_0  |
|python                    |3.7.5               | h359304d_0  |
|python-dateutil           |2.8.1                |      py_0  |
|pytz                      |2019.3                |     py_0  |
|pywavelets                |1.1.1            |py37h3b54f70_0   | 
|pyyaml                   | 5.1.2            |py37h1de35cc_0  |
|readline                |  7.0              |    h1de35cc_5  |
|scikit-image           |   0.15.0           |py37h0a44026_0  |
|scipy                 |    1.3.1            |py37h1a1e112_0  |
|setuptools           |     41.6.0            |       py37_0  |
|six                 |      1.12.0            |      py37_0  |
|sqlite             |       3.30.1             |  ha441bb4_0  |
|tensorboard       |        1.13.1           |py37haf313ee_0  |
|tensorflow       |         1.13.1         | mkl_py37h70c3834_0 | 
|tensorflow-base |          1.13.1         | mkl_py37h66b1bf0_0|  
|tensorflow-estimator   |   1.13.0         |            py_0  |
|termcolor             |    1.1.0          |          py37_1  |
|tk                   |     8.6.8           |     ha441bb4_0  |
|toolz               |      0.10.0          |           py_0   | 
|tornado            |       6.0.3           | py37h1de35cc_0  |
|werkzeug          |        0.16.0           |          py_0  |
|wheel            |         0.33.6         |          py37_0  |
|wrapt           |          1.11.2          | py37h1de35cc_0  |
|xz             |           5.2.4           |     h1de35cc_4  |
|yaml          |            0.1.7            |    hc338f04_2  |
|zlib         |             1.2.11           |    h1de35cc_3 | 
|zstd        |              1.3.7            |    h5bba6e5_0  |
