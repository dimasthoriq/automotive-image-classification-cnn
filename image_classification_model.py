#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[1]:


#Import needed libraries & packages
import h5py, cv2, io, os, splitfolders, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
# from google.colab import drive


# In[117]:


# # Splitting image folders into train, val, test folders (once via ipynb)
# input_folder = "C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\image"
# output = "C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\dataset"

# splitfolders.ratio(input_folder, output=output, seed=16, ratio=(.7, .15, .15))


# In[7]:


# # Mount google drive and set current directory
# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/KP CNN


# In[118]:


#Set directories path
base_dir = "C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\dataset"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

train_ertiga_dir = os.path.join(train_dir, 'Ertiga')
train_xl7_dir = os.path.join(train_dir, 'XL7')

validation_ertiga_dir = os.path.join(validation_dir, 'Ertiga')
validation_xl7_dir = os.path.join(validation_dir, 'XL7')

test_ertiga_dir = os.path.join(test_dir, 'Ertiga')
test_xl7_dir = os.path.join(test_dir, 'XL7')


# In[119]:


#Check total images per classes per sets
print('total training Ertiga images:', len(os.listdir(train_ertiga_dir)))
print('total training XL7 images:', len(os.listdir(train_xl7_dir)))

print('total validation Ertiga images:', len(os.listdir(validation_ertiga_dir)))
print('total validation XL7 images:', len(os.listdir(validation_xl7_dir)))

print('total testing Ertiga images:', len(os.listdir(test_ertiga_dir)))
print('total testing XL7 images:', len(os.listdir(test_xl7_dir)))


# In[2]:


# Preparing zero-valued Numpy array for cut objects
# Shape: image number, height, width, number of channels
x_train = np.zeros((1, 150, 150, 3))

# Preparing temp zero-valued Numpy array for current cut object
# Shape: image number, height, width, number of channels
x_temp = np.zeros((1, 150, 150, 3))

# Defining boolean variable to track arrays' shapes
first_object = True


os.chdir('C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\dataset\\train\\Ertiga')
# Showing currently active directory
print('Currently active directory:')
print(os.getcwd())
print()

for current_dir, dirs, files in os.walk('.', topdown = True):
    # Iterating all files
    for f in files:
        # Checking if filename ends with '.jpeg'
        if f.endswith('.jpeg'):
            image_array = cv2.imread(f)
            # Swapping channels from BGR to RGB by OpenCV function
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, (150, 150), interpolation=cv2.INTER_CUBIC)
            # Checking if it is the first object
            if first_object:
                # Assigning to the first position first object
                x_train[0, :, :, :] = image_array
                first_object = False
                # Collecting next objects into temp arrays
                # Concatenating arrays vertically
            else:
                # Assigning to temp array current object
                x_temp[0, :, :, :] = image_array
                # Concatenating vertically temp arrays to main arrays
                x_train = np.concatenate((x_train, x_temp), axis=0)


os.chdir('C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\dataset\\train\\XL7')
# Showing currently active directory
print('Currently active directory:')
print(os.getcwd())
print()

for current_dir, dirs, files in os.walk('.', topdown = True):
    # Iterating all files
    for f in files:
        # Checking if filename ends with '.jpeg'
        if f.endswith('.jpeg'):
            image_array = cv2.imread(f)
            # Swapping channels from BGR to RGB by OpenCV function
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, (150, 150), interpolation=cv2.INTER_CUBIC)
            # Checking if it is the first object
            x_temp[0, :, :, :] = image_array
            # Concatenating vertically temp arrays to main arrays
            x_train = np.concatenate((x_train, x_temp), axis=0)

print(x_train.shape)
os.chdir('C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN')


# In[121]:


# Magic function that renders the figure in a jupyter notebook
# instead of displaying a figure object
get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (8, 8)


# Defining a figure object with number of needed subplots
# ax is a (3, 5) numpy array
# To access specific subplot we call it by ax[0, 0]
figure, ax = plt.subplots(nrows=3, ncols=5)


# Plotting 15 examples along 3 rows and 5 columns
for i in range(3):
    for j in range(5):
        # Preparing random index
        ii = np.random.randint(low=0, high=x_train.shape[0])
        
        # Plotting current subplot
        ax[i, j].imshow(x_train[ii].astype('uint8'))
                
        # Hiding axis
        ax[i, j].axis('off')

# Adjusting distance between subplots
plt.tight_layout()

# Showing the plot
plt.show()


# # Data Preprocessing

# Let's set up data generators that will read pictures in our source folders, convert them to `float32` tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of 20 images of size 150x150 and their labels (binary).
# 
# As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the `[0, 1]` range (originally all values are in the `[0, 255]` range).
# 
# In Keras this can be done via the `keras.preprocessing.image.ImageDataGenerator` class using the `rescale` parameter. This `ImageDataGenerator` class allows you to instantiate generators of augmented image batches (and their labels) via `.flow(data, labels)` or `.flow_from_directory(directory)`. These generators can then be used with the Keras model methods that accept data generators as inputs: `fit_generator`, `evaluate_generator`, and `predict_generator`.

# In[13]:


train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1],
                                   featurewise_center=True, featurewise_std_normalization=True)
val_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True)

train_datagen.fit(x_train)
val_datagen.fit(x_train)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='binary')


# In[14]:


print(x_train.mean())
print(x_train.std())

print(train_datagen.mean)
print(train_datagen.std)


# # Eksperimen

# ## How Many ConvPool Pairs?

# ### Build Models to Evaluate 

# In[83]:


# Model ConvPool Pair (CPP) 1: RGB --> {8C3-P2} --> 128 --> 1
# Model ConvPool Pair (CPP) 2: RGB --> {8C3-P2} --> {16C3-P2} --> 128 --> 1
# Model ConvPool Pair (CPP) 3: RGB --> {8C3-P2} --> {16C3-P2} --> {32C3-P2} --> 128 --> 1
# Model ConvPool Pair (CPP) 4: RGB --> {8C3-P2} --> {16C3-P2} --> {32C3-P2} --> {64C3-P2} --> 128 --> 1

img_input = layers.Input(shape=(150, 150, 3))

# Building 1st CPP Model
x = layers.Conv2D(8, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model_cpp_1 = Model(img_input, output)
model_cpp_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# Building 2nd CPP Model
x = layers.Conv2D(8, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model_cpp_2 = Model(img_input, output)
model_cpp_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# Building 3rd CPP Model
x = layers.Conv2D(8, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model_cpp_3 = Model(img_input, output)
model_cpp_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# Building 4th CPP Model
x = layers.Conv2D(8, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model_cpp_4 = Model(img_input, output)
model_cpp_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model_cpp = [model_cpp_1, model_cpp_2, model_cpp_3, model_cpp_4]


# ### Train Built Models 

# In[41]:


print(" ")
print("TRAINING 1st Model")
print(" ")
history_cpp_1 = model_cpp_1.fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 2nd Model")
print(" ")
history_cpp_2 = model_cpp_2.fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 3rd Model")
print(" ")
history_cpp_3 = model_cpp_3.fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 4th Model")
print(" ")
history_cpp_4 = model_cpp_4.fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)


# ### Evaluate (Visualization) 

# In[45]:


# Accuracies of the 1st model
print('Model 1: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_cpp_1.history['acc']),
                                                                  max(history_cpp_1.history['val_acc']),
                                                                  max(history_cpp_1.history['loss']),
                                                                  max(history_cpp_1.history['val_loss'])))
print('Model 2: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_cpp_2.history['acc']),
                                                                  max(history_cpp_2.history['val_acc']),
                                                                  max(history_cpp_2.history['loss']),
                                                                  max(history_cpp_2.history['val_loss'])))
print('Model 3: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_cpp_3.history['acc']),
                                                                  max(history_cpp_3.history['val_acc']),
                                                                  max(history_cpp_3.history['loss']),
                                                                  max(history_cpp_3.history['val_loss'])))
print('Model 4: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_cpp_4.history['acc']),
                                                                  max(history_cpp_4.history['val_acc']),
                                                                  max(history_cpp_4.history['loss']),
                                                                  max(history_cpp_4.history['val_loss'])))


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_cpp_1.history['val_acc'], '-o')
plt.plot(history_cpp_2.history['val_acc'], '-o')
plt.plot(history_cpp_3.history['val_acc'], '-o')
plt.plot(history_cpp_4.history['val_acc'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4'], loc='lower right', fontsize='xx-large')

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)

# Giving name to the plot
plt.title('Models accuracies: Fine tuning ConvPool Pairs', fontsize=16)

# Showing the plot
plt.show()


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_cpp_1.history['val_loss'], '-o')
plt.plot(history_cpp_2.history['val_loss'], '-o')
plt.plot(history_cpp_3.history['val_loss'], '-o')
plt.plot(history_cpp_4.history['val_loss'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4'], loc='upper left')

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)

# Giving name to the plot
plt.title('Models loss: Fine tuning ConvPool Pairs', fontsize=16)

# Showing the plot
plt.show()


# ### Best Convolutional-Pooling Pairs: Model 4 (4 Pairs)
# 
# RGB --> {8C3-P2} --> {16C3-P2} --> {32C3-P2} --> {64C3-P2} --> 128 --> 1

# ## Feature Maps?

# ### Build Models to Evaluate 

# In[59]:


# Model FM 1: RGB --> {8C3-P2} --> {16C3-P2} --> {32C3-P2} --> {64C3-P2} --> 128 --> 1
# Model FM 2: RGB --> {16C3-P2} --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> 128 --> 1
# Model FM 3: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 128 --> 1
# Model FM 4: RGB --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> {512C3-P2} --> 128 --> 1
# Model FM 5: RGB --> {128C3-P2} --> {256C3-P2} --> {512C3-P2} --> {1024C3-P2} --> 128 --> 1

# Defining list to collect models in
model_FM = []

# Building models in a loop
for i in range(5):
  x = layers.Conv2D(8*(2**i), 3, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(16*(2**i), 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(32*(2**i), 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(64*(2**i), 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  temp = Model(img_input, output)
  temp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_FM.append(temp)


# ### Train Built Models 

# In[60]:


# Defining list to collect results in
history_FM = []

# Training models in a loop
for i in range(5):
    print(" ")
    print(f"TRAINING Model FM {i+1}")
    print(" ")
    temp = model_FM[i].fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)
    history_FM.append(temp)


# ### Evaluate (Visualization) 

# In[61]:


for i in range(5):
    print('Model FM {0}: MAX Training accuracy= {1:.5f}, MAX Validation accuracy= {2:.5f}, MAX Training loss= {3:.5f}, MAX Validation loss= {4:.5f}'.
                                                         format(i + 1,
                                                                max(history_FM[i].history['acc']),
                                                                max(history_FM[i].history['val_acc']),
                                                                max(history_FM[i].history['loss']),
                                                                max(history_FM[i].history['val_loss']),))


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_FM[0].history['val_acc'], '-o')
plt.plot(history_FM[1].history['val_acc'], '-o')
plt.plot(history_FM[2].history['val_acc'], '-o')
plt.plot(history_FM[3].history['val_acc'], '-o')
plt.plot(history_FM[4].history['val_acc'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)

# Giving name to the plot
plt.title('Models accuracies: Choosing the number of feature maps', fontsize=16)

# Showing the plot
plt.show()


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_FM[0].history['val_loss'], '-o')
plt.plot(history_FM[1].history['val_loss'], '-o')
plt.plot(history_FM[2].history['val_loss'], '-o')
plt.plot(history_FM[3].history['val_loss'], '-o')
plt.plot(history_FM[4].history['val_loss'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)

# Giving name to the plot
plt.title('Models losses: Choosing the number of feature maps', fontsize=16)

# Showing the plot
plt.show()


# ### Best number of feature maps combination:
# 
# RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 128 --> 1

# ## How Many Neurons in Output Layers?

# ### Build Models to Evaluate 

# In[64]:


# Model Neurons in Output Layers (NOL) 1: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 128 --> 1
# Model NOL 2: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 256 --> 1
# Model NOL 3: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 512 --> 1
# Model NOL 4: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 1024 --> 1
# Model NOL 5: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 2048 --> 1

model_nol = []

for i in range(5):
  x = layers.Conv2D(32, 3, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(64, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(128, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(256, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(128*(2**i), activation='relu')(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  temp = Model(img_input, output)
  temp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_nol.append(temp)


# ### Train Built Models 

# In[65]:


# Defining list to collect results in
history_nol = []

# Training models in a loop
for i in range(5):
  print(" ")
  print(f"TRAINING Model NOL {i+1}")
  print(" ")
  temp = model_nol[i].fit(train_generator, steps_per_epoch=17, epochs=15, validation_data=validation_generator, validation_steps=4, verbose=1)
  history_nol.append(temp)


# ### Evaluate (Visualization) 

# In[67]:


for i in range(5):
    print('Model NOL {0}: MAX Training accuracy= {1:.5f}, MAX Validation accuracy= {2:.5f}, MAX Training loss= {3:.5f}, MAX Validation loss= {4:.5f}'.
          format(i + 1, max(history_nol[i].history['acc']), max(history_nol[i].history['val_acc']),
                 max(history_nol[i].history['loss']), max(history_nol[i].history['val_loss'])))


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_nol[0].history['val_acc'], '-o')
plt.plot(history_nol[1].history['val_acc'], '-o')
plt.plot(history_nol[2].history['val_acc'], '-o')
plt.plot(history_nol[3].history['val_acc'], '-o')
plt.plot(history_nol[4].history['val_acc'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)

# Giving name to the plot
plt.title('Models accuracies: Choosing the number of neutrons in output layer', fontsize=16)

# Showing the plot
plt.show()


# In[69]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_nol[0].history['val_loss'], '-o')
plt.plot(history_nol[1].history['val_loss'], '-o')
plt.plot(history_nol[2].history['val_loss'], '-o')
plt.plot(history_nol[3].history['val_loss'], '-o')
plt.plot(history_nol[4].history['val_loss'], '-o')

# Showing legend
plt.legend(['model_1', 'model_2', 'model_3', 'model_4', 'model_5'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)

# Giving name to the plot
plt.title('Models losses: Choosing the number of neutrons in output layer', fontsize=16)

# Showing the plot
plt.show()


# ### Best number of neurons in output layer: 1024 (Model 4)
# 
# RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 1024 --> 1

# ## Best Preprocessing Method?

# ### Build Generator to Evaluate 

# In[77]:


# Generator 1: Pixel Scaling with normalization (1/255)
# Generator 2: Pixel scaling with centering (- mean image)
# Generator 3: Pixel scaling with standardizing (/ std image)

#Building generator 1
train_datagen_1 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1])
val_datagen_1 = ImageDataGenerator(rescale=1./255)

train_generator_1 = train_datagen_1.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator_1 = val_datagen_1.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='binary')



#Building generator 2
train_datagen_2 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1], featurewise_center=True)
val_datagen_2 = ImageDataGenerator(rescale=1./255, featurewise_center=True)

train_datagen_2.fit(x_train)
val_datagen_2.fit(x_train)

train_generator_2 = train_datagen_2.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator_2 = val_datagen_2.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='binary')



#Building generator 3
train_datagen_3 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1],
                                     featurewise_center=True, featurewise_std_normalization=True)
val_datagen_3 = ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True)

train_datagen_3.fit(x_train)
val_datagen_3.fit(x_train)

train_generator_3 = train_datagen_3.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator_3 = val_datagen_3.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='binary')


# In[84]:


#Building model: RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 1024 --> 1
model_gen = []
for i in range(3):
  x = layers.Conv2D(32, 3, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(64, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(128, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(256, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(1024, activation='relu')(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  temp = Model(img_input, output)
  temp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_gen.append(temp)


# ### Train Built Models 

# In[85]:


print(" ")
print(f"TRAINING WITH NORMALIZED DATA")
print(" ")
history_gen_1 = model_gen[0].fit(train_generator_1, steps_per_epoch=17, epochs=30, validation_data=validation_generator_1, validation_steps=4, verbose=1)

print(" ")
print(f"TRAINING WITH CENTRALIZED DATA")
print(" ")
history_gen_2 = model_gen[1].fit(train_generator_2, steps_per_epoch=17, epochs=30, validation_data=validation_generator_2, validation_steps=4, verbose=1)

print(" ")
print(f"TRAINING WITH STANDARDIZED DATA")
print(" ")
history_gen_3 = model_gen[2].fit(train_generator_3, steps_per_epoch=17, epochs=30, validation_data=validation_generator_3, validation_steps=4, verbose=1)


# ### Evaluate (Visualization) 

# In[86]:


# Accuracies of the 1st model
print('Normalized Data: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_gen_1.history['acc']),
                                                                  max(history_gen_1.history['val_acc']),
                                                                  max(history_gen_1.history['loss']),
                                                                  max(history_gen_1.history['val_loss'])))
print('Centralized Data: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_gen_2.history['acc']),
                                                                  max(history_gen_2.history['val_acc']),
                                                                  max(history_gen_2.history['loss']),
                                                                  max(history_gen_2.history['val_loss'])))
print('Standardized Data: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Max Training loss= {2:.5f}, Max Validation loss= {3:.5f}'.
                                                           format(max(history_gen_3.history['acc']),
                                                                  max(history_gen_3.history['val_acc']),
                                                                  max(history_gen_3.history['loss']),
                                                                  max(history_gen_3.history['val_loss'])))


# In[87]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_gen_1.history['val_acc'], '-o')
plt.plot(history_gen_2.history['val_acc'], '-o')
plt.plot(history_gen_3.history['val_acc'], '-o')

# Showing legend
plt.legend(['Normalized', 'Centralized', 'Standardized'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)

# Giving name to the plot
plt.title('Models accuracies: Best pixel-scaling methods', fontsize=16)

# Showing the plot
plt.show()


# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (12.0, 6.0)

# Plotting accuracies for every model
plt.plot(history_gen_1.history['val_loss'], '-o')
plt.plot(history_gen_2.history['val_loss'], '-o')
plt.plot(history_gen_3.history['val_loss'], '-o')

# Showing legend
plt.legend(['Normalized', 'Centralized', 'Standardized'])

# Giving name to axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)

# Giving name to the plot
plt.title('Models losses: Best pixel-scaling methods', fontsize=16)

# Showing the plot
plt.show()


# ### Best pixel-scaling method: 
# 
# Standardizing (feature-wise)

# ## Best Model Architecture:
# 
# RGB --> {32C3-P2} --> {64C3-P2} --> {128C3-P2} --> {256C3-P2} --> 1024 --> 1
# 
# With feature-wised standardized data ((x/255 - avg)/std)

# # Data Modeling & Processing

# Because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

# Next, we'll configure the specifications for model training. We will train our model with the `binary_crossentropy` loss, because it's a binary classification problem and our final activation is a sigmoid. (For a refresher on loss metrics, see the [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture).) We will use the `adam` optimizer. During training, we will want to monitor classification accuracy.
# 
# **NOTE**: In this case, using the [RMSprop optimization algorithm](https://wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp) is preferable to [stochastic gradient descent](https://developers.google.com/machine-learning/glossary/#SGD) (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as [Adam](https://wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [Adagrad](https://developers.google.com/machine-learning/glossary/#AdaGrad), also automatically adapt the learning rate during training, and would work equally well here.)

# ## Building Model

# In[132]:


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 32 filters that are 3x3, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 64 filters that are 3x3, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 128 filters that are 3x3, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Fourth convolution extracts 256 filters that are 3x3, convolution is followed by max-pooling layer with a 2x2 window  
x = layers.Conv2D(256, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers  
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 1024 hidden units
x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully connected layer + sigmoid output layer
model = Model(img_input, output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the feature map.

# ## Training

# In[133]:


history = model.fit(
      train_generator,
      steps_per_epoch=17,  # 265 images = batch_size * steps
      epochs=50,
      validation_data=validation_generator,
      validation_steps=4,  # 57 images = batch_size * steps
      verbose=2, callbacks=[es, mc])


# ## Visualizing Intermediate Representations
# 
# To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.
# 
# Let's pick a random image from the training set, and then generate a figure where each row is the output of a layer, and each image in the row is a specific filter in that output feature map. Rerun this cell to generate intermediate representations for a variety of training images.

# In[23]:


# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
train_ertiga_fnames = os.listdir(train_ertiga_dir)
train_xl7_fnames = os.listdir(train_xl7_dir)

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random image from the training set.
ertiga_img_files = [os.path.join(train_ertiga_dir, f) for f in train_ertiga_fnames]
xl7_img_files = [os.path.join(train_xl7_dir, f) for f in train_xl7_fnames]
img_path = random.choice(ertiga_img_files + xl7_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# As you can see we go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called "sparsity." Representation sparsity is a key feature of deep learning.
# 
# 
# These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. You can think of a convnet (or a deep network in general) as an information distillation pipeline.

# ## Evaluating Accuracy and Loss for the Model
# 
# Let's plot the training/validation accuracy and loss as collected during training:

# In[144]:


# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc'][:19]
val_acc = history.history['val_acc'][:19]

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss'][:19]
val_loss = history.history['val_loss'][:19]

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.5])
plt.title('Training and validation loss')

print('Best Model: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Min Training loss= {2:.5f}, Min Validation loss= {3:.5f}'
      .format(max(acc), max(val_acc),
              min(loss), min(val_loss)))


# ## Testing

# In[136]:


# load the saved model
saved_model = load_model('best_model.h5')


# In[137]:


# load the test dataset
test_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True)
test_datagen.fit(x_train)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=1,
    shuffle = False,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

filenames = test_generator.filenames
nb_samples = len(filenames)


# In[138]:


#Predicting the test dataset using saved model
pred = model.predict(test_generator, steps = nb_samples)


# In[139]:


y_true = test_generator.classes
y_pred = []
for i in pred:
    if i <= 0.5:
        y_pred.append(0)
    else:
        y_pred.append(1)


# In[140]:


print(classification_report(y_true, y_pred))

c_m = confusion_matrix(y_true, y_pred)
print("Confusion Matrix: ")
print(c_m)


# In[141]:


labels = ['Ertiga', 'XL7']

plt.rcParams['figure.figsize'] = (12,6)

display_c_m = ConfusionMatrixDisplay(c_m, display_labels=labels)

# Plotting confusion matrix
# Setting colour map to be used
display_c_m.plot(cmap='OrRd')
# Other possible options for colour map are: 'autumn_r', 'Blues', 'cool', 'Greens', 'Greys', 'PuRd', 'copper_r'

# Giving name to the plot
plt.title('Confusion Matrix: Test Dataset')

# Showing the plot
plt.show()


# # Clean Up
# 
# Before running the next exercise, run the following cell to terminate the kernel and free memory resources:

# In[ ]:


import os, signal
os.kill(os.getpid(), signal.SIGKILL)

