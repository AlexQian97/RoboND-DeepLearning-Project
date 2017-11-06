# RoboND Deep Learning Project

### Goal

Build a deep neural network to identify and track a target in simulation.

### Step 1: build a fully convolutional network (FCN)
This image shows the structure of fully convolutional network.

![fcn](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/fcn.png)

There are a few techniques used in the network
### Encoder
The encoder layers are used for extracting features from the input images. More layers will help the network
learn higher level features. The separable covolutional layer is used, which reduces the number of parameters needed,
thus increasing efficiency for the network.

### 1x1 convolution
In normal convolutional neural networks, the last layer is a fully connected layer, which flatten the input into a
2-dimensional tensor. Such operation would lose the spatial information.

As a result, the 1x1 convolution layer is used when we want to reduce the dimensionality without loss of spatial information.
In other words, it is a cheap way to make the network deeper, without too much cost of computation. In addition, the input
image can be of any dimension because of this layer.

This layer is used after normal convolution operations.

### Decoder
The decoder layers are used to upscale the output of the encoder layers to have a image of the same size as the original
image. Therefore, prediction of each pixel in the original image can be obtained. However, when we decode the output of 
the image back to the original image size, some information is potentially lost. To retain the information, skip connections
are used.

### Skip connections
The skip connections allow the network use information from different resolution scales. As a result, the network can make
more precise segmentation decision.

The model finally be used is shown here.
```python
def fcn_model(inputs, num_classes):
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    layer_1 = encoder_block(inputs, 32, 2)
    layer_2 = encoder_block(layer_1, 64, 2)
    layer_3 = encoder_block(layer_2, 128, 2)
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    layer_4 = conv2d_batchnorm(layer_3, 256, kernel_size=1, strides=1)
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    layer_5 = decoder_block(layer_4, layer_2, 128)
    layer_6 = decoder_block(layer_5, layer_1, 64)
    x = decoder_block(layer_6, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```
where `encoder_block` is just one separable convolutional layer while the `decoder_block` has 3 separable convolutional 
layer, which can help the network learns in my experiment. The improvement is especially useful when dealing with false 
positive. However, there are no objects like car or animal in the training set. As a result, this network should perform
terribly when dealing with new environments. To adapt to new environments, new training data is needed. If the result is 
not good enough, deeper network might be required.

A terrible stretch is attached.
![fcn_structure](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/fcn_structure.png)

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
1. learning rate
Learning rate determines the speed of learning. Higher learning rate can lead to faster convergence but lower the final 
accuracy. The lower learning rate can help the network get a good final result. `learning_rate = 0.01` was used in my 
first experiment. However, the network is undertrained. As a result, I lowered it to 0.005.
2. batch_size
The batch_size is the number of inputs passed into the network at a time.
The batch_size should be as high as possible under the condition that the machine has enough memory. 32 is set for my 
laptop and I forget to change it when training on the GPU instance.
3. num_epochs
The num_epochs determines number of times the training data is passed into the network. I changed it from 10 to 100 for 
more training with lower learning rate.
4. steps_per_epoch
The number of batches is determined by the total number of training images and the batch size.
5. validation_steps
Same thing, but for validation images this time
6. workers
Maximum number of processes to spin up. It could be set higher when training on the GPU instance as the hardware should 
be good.

###results
The learning curve is shown below.

![learning curve](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/learning_curve.png)

Some test images are attached as well.

The network can identify the target in close range.

![result_1](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/result_1.png)

Few false positive.

![result_2](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/result_2.png)

The network can identify the target from far away.

![result_3](https://github.com/AlexQian97/RoboND-DeepLearning-Project/blob/master/images/result_3.png)


The final score is 0.521(52.1%), which is better than the minimum accuracy.

### Future enhancements
More data collection can definitely help the network to learn better. However, it will require more computational 
resources to train the network.

Three encoder and decoder blocks are used in this network, more blocks can enable higher level feature learning.
