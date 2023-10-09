# Root Diameter Prediction with Residual Neural Network
![ResNet](https://github.com/Computational-Plant-Science/Root-Diameter-Prediction-with-Residual-Neural-Network/assets/133724174/3bf17fb4-86a3-4c52-9e69-5c7f1f1bf721)
ResNet Neural Network

# Introduction
Characterizing root systems in field conditions is a challenging task. Surprisingly, not many people have started using Fiber Optic (FO) sensors for this purpose. Utilizing FO sensors for root system characterization could be very helpful for plant breeders aiming to develop stress-resilient crop varieties. To achieve this, we collected data using three FO sensors and analyzed it using a neural network. This code is used to predict the diameter of plant roots using a residual neural network.

# Network Architecture
The input consists of a 2 x 2 convolutional layer, followed by batch normalization, an activation function (ReLU), and a max-pooling layer. The convolutional layer comprises 16 feature maps. The output of the max-pooling layer is connected to 21 residual blocks, each composed of two 5 x 5 convolutional layers, followed by two batch normalizations and an activation functions (ReLU). Each convolutional layer consists of 16 feature maps. The output of the final residual block is connected to three dense layers with values of 50, 40, and 20, respectively, and finally, an output layer. 

In this setup, "d" denotes the sample size, set at 300, while "n" represents the number of channels, fixed at 2, and "m" represent the number of classes, which is also set at 2. Notably, each dense layer is intricately connected with a corresponding dropout layer with a dropout rate of 0.3, optimizing the network's performance.

# Data Sets and Preparations
For model development we have generated artificial data by simulation. As shown in the above figure, two sensors were positioned on either side of the location where the rod was inserted into the soil, while the third sensor was placed directly under the rod. Data was captured as the rod was pressed into the soil for a total of 12 minutes, reaching a depth of 15cm. Two different diameter rods were used in the experiments. A total of 15 trials of data were collected for each rod, resulting in a total of 30 data trials.

In each data trial, we took the signals and divided them into smaller parts, each having a size of 300. We did this carefully for every trial, and it gave us a total of 3228 pieces of data. Out of these, we used 2421 for training and the remaining 807 samles for testing our system. This way, we made sure to have a good mix of data for training and testing the model.


# How to Run the script  
Run the main.py to train the model, and it will automatically load the data from Excel files, start training, and execute utils.py. Additionally, the model's performance results, including accuracy, recall, and precision, will be displayed.

# Model Training and Evaluation Results
The model was trained for 300 epochs, and after this training, We tested the model with a total of 807 samples, and it demonstrated an impressive test accuracy of 91.07%. These results are quite promising, indicating that the model is performing well. Moreover, as we continue to increase the number of samples for testing, we anticipate further improvements in the model's performance. This suggests that with more data, we can refine and enhance the model's accuracy and effectiveness.

# Requirements
seaborn, matplotlib, tensorflow, tensorflow_docs, sklearn etc.
