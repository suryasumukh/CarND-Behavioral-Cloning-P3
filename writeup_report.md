# **Behavioral Cloning**  

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution.png "distribution"
[image2]: ./examples/preprocessing.png "preprocessing"
[image3]: ./examples/sampling.png "sampling"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Project includes the following files:
* model.py containing the script to create and train the model
* utils.py containing keras data generator functions and other utility functions  
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

#### An appropriate model architecture has been employed

As a starting point LeNet architecture was chosen. After trying to tune the network parameters for some time, the model learned to drive well on a straight lane but did not do so well with curves. Next, the [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) model was employed. The model did not work straight off the box with default settings. After modifying the network hyper-parameters, the model was able to drive around track 1 with out leaving the track. However it goes very close to the lane lines but does not go over or on it. Data for both approaches used the sample training data provided by Udacity. More on dataset is discussed in Q3. The code for the comm.ai model can be found in `model.py` lines 38-60.

#### Attempts to reduce overfitting in the model

Comma.ai model uses two dropout layers with 0.2 and 0.5 drop probabilities. The first dropout is applied after the convolution stack and the second is applied to the fully connected layer with 256 units.

To further reduce overfitting, the data is divided in train, validation and test sets. The sample data is skewed with most steering angles centered around 0.

![alt text][image1]  

To reduce the frequency of these steering angles, training instances are randomly sampled and absolute value of steering angles in the range 0 to 0.1 and 0.15 to 0.3 are accepted with a probability of 20%. The result of such sampling is still skewed, not completely uniform either; but reduces the frequency of angles centered around 0. Random shear was tried but the network it did not seem to perform significantly any better. The following is the histogram of the sampling result.

![alt text][image3]

For all experiments, the provided sample data was used. Train set was augmented with vertically flipping the images. The generator are described in Q3. The sampling code can be found in `utils.py` lines 74-90.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 57). The following are the values of parameters that performed well on the validation set and the simulator.
* epochs: 10
* learning rate: 0.001  
* batch size: 128  
* angle correction: 0.2  
* input shape: 70, 320, 1  
The validation mse is around ~0.0136 for these parameters, train mse was around ~0.0221 and test mse was ~0.0122.
The training is started with `epoch=10`, the early stopping callback usually stops between 6-8 epochs.


#### 2. Final Model Architecture

The final model architecture (model.py lines 38-60) is an adaptation of comma.ai model. The model takes in a image with shape (70, 320, 1). The model uses three convolution layers. The first convolution layers has 8 kernels with kernel shape (8, 8) and strides with (4, 4). The second layer has 16 kernels with shape (5, 5) and stride (2, 2). The final convolution layers has 32 kernels with shape (5, 5) and stride (2, 2). All the convolution layers use `padding=same`. The output of the convolution block is flattened to get a vector of shape 3200. Then a dropout of 0.2 reject probability is applied and fed to a fully connected layer with 256 units. Another dropout of 0.5 reject probability is applied here. Like the original comma.ai model, all activation have been ELU except for the output unit which is a linear neuron.

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 70, 320, 1)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 18, 80, 8)         520       
_________________________________________________________________
elu_1 (ELU)                  (None, 18, 80, 8)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 40, 16)         3216      
_________________________________________________________________
elu_2 (ELU)                  (None, 9, 40, 16)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 20, 32)         12832     
_________________________________________________________________
flatten_1 (Flatten)          (None, 3200)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 3200)              0         
_________________________________________________________________
elu_3 (ELU)                  (None, 3200)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               819456    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
elu_4 (ELU)                  (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 836,281
Trainable params: 836,281
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

The data for all experiments came from the sample data provided in the course. The steering angles are mostly centered around zero and fall between [-1, 1]. This set is divided into train set (80%), validation set (10%) and test set (10%) randomly with `sklearn.model_selection.train_test_split` function. The code for this can be found in `model.py` in the function `main` lines 68-70.

Keras generators objects are created from train, validation and test sets. The code can be found in `utils.py`. For training, a class `ImageGenerator` a subclass of `keras.utils.Sequence` is defined. Data is augmented with flip images of the train set. Essentially now train set is twice as large. In addition, left and right camera images are used and for these images a correction of 0.2 is applied. The center image steering angle is left unchanged. For every mini batch of an epoch, images are randomly samples with constrains on angles between 0-0.1 and 0.15-3 and chosen only 20% of the time to reduce the skew. The images are read with openCV library. Since we are mostly concerned with lanes and driving in between the lane lines, images are converted to HLS color space and the S channel is considered, having clearly picking up lane demarkations. The image is then cropped to eliminate the top half which contains images of sky and surroundings. The bottom 20 pixels are also discarded since it mostly consists of the car hood. The preprocessed image shape is (70, 320, 1). The network is trained with mini batches of such images. The code for this class definiation can be found in `utils.py` lines 37-95. The following are images of train sample in HLS space.

![alt text][image2]

ModelCheckpoint and EarlyStopping callbacks are used to keep track of best performing weights and saving the model. The patience parameter is set to 2 for early stopping. This means that if validation accuracy fails to improve after two successive epochs, the training is terminated.

The validation and test generators produce only center camera images. The code for this is defined in a class called `CenterImageGenerator` in `utils.py` lines 97-135. The network is trained with above described hyper params.

#### Video

Here's the link to the video output - [test_drive.py](./test_drive.mp4)
