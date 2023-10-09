"""
Author         : Kabir Hossain, School of Plant Science, The University of Arizona.
Date           : 2023-09-28

"""
# %%
import pathlib
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from keras import layers, losses, metrics, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from sklearn.metrics import accuracy_score
from utils import *
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
# %%
# Funnction define
# %%

def processData(Y,L,bothsensor):
	trainX=[]
	labels=[]

	for i in range(0,len(Y)*bothsensor,bothsensor):
		# print(i)
		dataX = ast.literal_eval(Y[i])
		new_dataX = [[i] for i in dataX]
		trainX.append(new_dataX)
		label = L[i]
		labels.append(label)
	trainX=np.array(trainX)
	trainY=np.array(labels)

	return trainX, trainY

def resnet(num_classes,num_tran,n):
	kernel_size = (5, 5)
	n_filter = 16
	dropout_rate=0.3
	input = layers.Input(shape=(num_tran, n), name="Input")
	x = layers.Reshape((num_tran, n, 1))(input)
	x = layers.Conv2D(n_filter, 2, padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)

	# n_filter = 2 * n_filter
	x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)

	# n_filter = 2 * n_filter
	x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)

	x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)

	x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)
	x = resblock(x, kernel_size, filters=n_filter)

	# x = layers.Flatten()(x)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(50, activation="relu", kernel_regularizer='l2')(x)
	x = layers.Dropout(dropout_rate)(x)
	x = layers.Dense(40, activation="relu", kernel_regularizer='l2')(x)
	x = layers.Dropout(dropout_rate)(x)
	x = layers.Dense(30, activation="relu", kernel_regularizer='l2')(x)
	output = layers.Dense(num_classes)(x)
	model = models.Model(input, output, name="ResNet30RealData")
	model.summary()
	return model

def get_callbacks(name):
	name1 = name + '/log.csv'
	return [
		tfdocs.modeling.EpochDots(),
		tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy',	patience=300,min_delta=1e-3	), # original
		tf.keras.callbacks.TensorBoard(Path(logdir, name)),
		tf.keras.callbacks.CSVLogger(Path(logdir, name1)),
	]

def compile_and_fit(model, name,x_train,y_train,los, batch_size, max_epochs=10000):
	learning_rate = 1e-3
	lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=5000, decay_rate=1, staircase=False)
	optimizer = tf.keras.optimizers.Adam(lr_schedule)
	model.compile(optimizer=optimizer,loss=los,metrics=[metrics.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy'),"accuracy"]) # original
	# model.compile(optimizer=optimizer,loss=los,metrics=[metrics.SparseCategoricalCrossentropy(from_logits=True, name='val_accuracy'),"accuracy"])
	model.summary()
	model.fit(x_train,y_train,epochs=max_epochs,batch_size=batch_size,validation_split=0.2,callbacks=get_callbacks(name))


# load the real dataset and process

# Read CSV file
filename='Z:\\Works Kabir\\Scripts\\HelloWorld\\Signal_AnalysisV6.csv'
df = pd.read_csv(filename)
Y=df.Signal_Y
Y2=df.Signal_X
L=df.label

bothsensor=1 #[1 for yes and 2 for no]
trainX1, trainY = processData(Y,L,bothsensor)
trainX2, trainY2 = processData(Y2,L,bothsensor)
print(trainX1.shape)
print(trainY.shape)


trainY=trainY
trainY=np.array(trainY)

trainX = np.concatenate([trainX1, trainX2], axis=2)

# Training and Testing split

X_train, X_test, y_train, y_test=train_test_split(trainX,trainY,test_size=0.25,random_state = 42,shuffle=True)

# %%
############## plot one case ################
colors = [c for c, _ in mcolors.TABLEAU_COLORS.items()]
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
for i, ax in enumerate(axs.flat):
	ax.plot(X_train[0, :, i], colors[i])
plt.show()



# shuffle the datasets
label_train = y_train
label_test = y_test

ind_train = np.random.permutation(X_train.shape[0])
ind_test = np.random.permutation(X_test.shape[0])

x_train = X_train[ind_train, :, :]
x_test = X_test[ind_test, :, :]
y_train = label_train[ind_train]
y_test = label_test[ind_test]

# %% get the working path
current_file = Path(__file__).stem
print(current_file)
model_name = current_file
logdir = Path("tensorboard_logs", "Trial")

# %% parameter settings
x_test = np.transpose(x_test, (0, 2, 1))
x_train = np.transpose(x_train, (0, 2, 1))

############### Trainining start here
# Parameter define
epochs = 300
batch_size = 24 #24
n = x_train.shape[-1]
# num_classes = len(counts)
# num_classes = len(dTye) # for artificial data I created
num_classes = 2
num_tran = X_train.shape[-1]
model_path = Path(logdir, model_name, 'model/model_diameter.h5')
# Model Define
model = resnet(num_classes,num_tran,n)
# Model Compile and training
compile_and_fit(model,model_name,x_train,y_train,losses.SparseCategoricalCrossentropy(from_logits=True),batch_size,max_epochs=epochs)

# Save the model
model.save(model_path)

#Prediction
model_pred = model.evaluate(x_test, y_test, verbose=2)
y_prob = np.max(model.predict(x_test), axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

# Model Evaluation
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Test accuracy",accuracy_score(y_test, y_pred))
print("Confusion matrix",confusion_mtx)
print("Prediction",y_pred)
print("Actual",y_test)
print('Precision: ', precision)
print('Recall: ', recall)


