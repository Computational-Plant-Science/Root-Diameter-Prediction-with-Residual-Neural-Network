# Root Diameter Prediction with Residual Neural Network
Examples of images along with their corresponding ground truth and predicted segmented images.
![image](https://github.com/Computational-Plant-Science/Root_Image_Segmentation/assets/133724174/987f30db-9f33-45f1-8214-75c19554644b)

# Data
Roots were grown in a mesh frame that supported the root system after removing the soil. Then the images were captured using a camera at the University of Georgia. 

The data can be found in the folder "data/root_data."

# Data Augmentation
For the initial experiment, the training data consists of 30 3456x5184 images, which were augmented using ImageDataGenerator to feed a deep learning neural network. However, here are 10 images (a total of 20 images including original and mask images) that have been published for testing purposes.

For more details, refer to preProcess.py.

# How to Run the script 

Run the main.py to train the model, and it automatically executed preProcess.py for preprocessing the images before training. Additionally, I created a separate script named preTrainedModel.py to test images using the trained model.

# Model
This deep neural network is implemented using the Keras functional API, allowing easy experimentation with various architectures.

# Training
The model was trained for 5 epochs, and after this training, the training accuracy is more than 0.98.

The loss function used during training is binary crossentropy. 

# Requirements
keras, tensorflow, scikit-image, opencv-python and matplotlib 
