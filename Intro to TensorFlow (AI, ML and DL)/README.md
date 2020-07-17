# Intro to TensorFlow (AI, ML and DL)
#### Course: [Coursera](https://www.coursera.org/learn/introduction-tensorflow)

### Main Takeaway

#### Week1: We learn about the basic flow of a tensor model.
> This is the basic flow:
```python3
model = keras.Sequential()
model.compile(optimizer, loss)
model.fit(x, y, epochs)
```

#### Week2: We now look into a lot more detail when it comes to designing a NN.
> Taking input.
```python3
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
x_train, x_test = x_train/255.0, x_test/255.0
```
> Making a custom callback to **stop training**.
```python3
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):   # 99% accuracy
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
                
callbacks = myCallback()    # class object
```
> Adding several layers(3 layer NN)
```python3
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
```
> Calling the callback in the fit function.
```python3
model.fit(x_train, y_train, epochs = 10, callbacks=[callbacks])
```

#### Week3: Time for Convolutions.
> Adding a convolution layer and then MaxPooling it, before the inputs.
```python3
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
```

#### Week4: Convolutions on complex images.
> ImageDataGenerator is a really handy tool.
```python3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
```
> flow_from_directory() returns (x=\[batch of images with shape (batch_size, *target_size, channels)\], y=\[corresponding labels\])
```python3
train_generator = train_datagen.flow_from_directory(
                "/tmp/h-or-s",
                target_size=(150, 150),
                class_mode='binary')

history = model.fit_generator(
                train_generator,
                epochs=30,
                callbacks=[callbacks])
```
