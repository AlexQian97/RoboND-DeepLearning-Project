# RoboND Deep Learning Project

### Goal

Build a deep neural network to identify and track a target in simulation.

### Step 1: build a fully convolutional network (FCN)

The model finally be used is shown here.
```python
def fcn_model(inputs, num_classes):
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    layer_1 = encoder_block(inputs, 32, 2)
    layer_2 = encoder_block(layer_1, 64, 2)
    layer_3 = encoder_block(layer_2, 128, 2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    layer_4 = conv2d_batchnorm(layer_3, 256, kernel_size=1, strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    layer_5 = decoder_block(layer_4, layer_2, 128)
    layer_6 = decoder_block(layer_5, layer_1, 64)
    x = decoder_block(layer_6, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```
where `encoder_block` is just one sepearable convolutional layer while the `decoder_block` has 3 sepearable convolutional 
layer, which can help the network learns in my experiment.

### Step 2: collect more data
Several runs of data are collected by the guidance. After preprocessing, they are placed in `/train` and `/validation`.

### Step 3: parameter tuning
The parameters used in submission is attached.
```python
learning_rate = 0.005
batch_size = 32
num_epochs = 100
steps_per_epoch = 100
validation_steps = 50
workers = 4
```
This set of parameters comes from my previous experiment where the network is undertrained. I lowered the learning rate
and add much more episodes.

###results
The learning curve is shown below.
[!learning curve](https://github.com/AlexQian97/RoboND-DeepLearning-Project/images/learning_curve.png)

Some test images are attached as well.

The network can identify the target in close range.
[!result_1](https://github.com/AlexQian97/RoboND-DeepLearning-Project/images/result_1.png)

Few false positive.
[!result_2](https://github.com/AlexQian97/RoboND-DeepLearning-Project/images/result_2.png)

The network can identify the target from far away.
[!result_3](https://github.com/AlexQian97/RoboND-DeepLearning-Project/images/result_3.png)


The final score is 0.521(52.1%), which is better than the minimum accuracy.
