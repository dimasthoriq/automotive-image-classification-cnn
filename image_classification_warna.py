#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[2]:


#Import needed libraries & packages
import h5py, cv2, io, os,  random
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
from google.colab import drive


# In[ ]:


# # Splitting image folders into train, val, test folders (once via ipynb)
# input_folder = "C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\warna"
# output = "C:\\Users\\Dimas Ahmad\\Downloads\\KP CNN\\dataset warna"

# splitfolders.ratio(input_folder, output=output, seed=16, ratio=(.7, .15, .15))


# In[3]:


# Mount google drive and set current directory
# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/KP CNN


# In[4]:


#Set directories path
base_dir = "/content/drive/My Drive/Colab Notebooks/KP CNN/dataset warna"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

train_0_dir = os.path.join(train_dir, 'Black')
train_1_dir = os.path.join(train_dir, 'Burgundy')
train_2_dir = os.path.join(train_dir, 'Grey')
train_3_dir = os.path.join(train_dir, 'Khaki')
train_4_dir = os.path.join(train_dir, 'Orange')
train_5_dir = os.path.join(train_dir, 'Red')
train_6_dir = os.path.join(train_dir, 'Silver')
train_7_dir = os.path.join(train_dir, 'Two tone')
train_8_dir = os.path.join(train_dir, 'White')

validation_0_dir = os.path.join(validation_dir, 'Black')
validation_1_dir = os.path.join(validation_dir, 'Burgundy')
validation_2_dir = os.path.join(validation_dir, 'Grey')
validation_3_dir = os.path.join(validation_dir, 'Khaki')
validation_4_dir = os.path.join(validation_dir, 'Orange')
validation_5_dir = os.path.join(validation_dir, 'Red')
validation_6_dir = os.path.join(validation_dir, 'Silver')
validation_7_dir = os.path.join(validation_dir, 'Two tone')
validation_8_dir = os.path.join(validation_dir, 'White')

test_0_dir = os.path.join(test_dir, 'Black')
test_1_dir = os.path.join(test_dir, 'Burgundy')
test_2_dir = os.path.join(test_dir, 'Grey')
test_3_dir = os.path.join(test_dir, 'Khaki')
test_4_dir = os.path.join(test_dir, 'Orange')
test_5_dir = os.path.join(test_dir, 'Red')
test_6_dir = os.path.join(test_dir, 'Silver')
test_7_dir = os.path.join(test_dir, 'Two tone')
test_8_dir = os.path.join(test_dir, 'White')


# In[5]:


#Check total images per classes per sets
print('total training Black images:', len(os.listdir(train_0_dir)))
print('total training Burgundy images:', len(os.listdir(train_1_dir)))
print('total training Grey images:', len(os.listdir(train_2_dir)))
print('total training Khaki images:', len(os.listdir(train_3_dir)))
print('total training Orange images:', len(os.listdir(train_4_dir)))
print('total training Red images:', len(os.listdir(train_5_dir)))
print('total training Silver images:', len(os.listdir(train_6_dir)))
print('total training Two tone images:', len(os.listdir(train_7_dir)))
print('total training White images:', len(os.listdir(train_8_dir)))
print(' ')

print('total validation Black images:', len(os.listdir(validation_0_dir)))
print('total validation Burgundy images:', len(os.listdir(validation_1_dir)))
print('total validation Grey images:', len(os.listdir(validation_2_dir)))
print('total validation Khaki images:', len(os.listdir(validation_3_dir)))
print('total validation Orange images:', len(os.listdir(validation_4_dir)))
print('total validation Red images:', len(os.listdir(validation_5_dir)))
print('total validation Silver images:', len(os.listdir(validation_6_dir)))
print('total validation Two tone images:', len(os.listdir(validation_7_dir)))
print('total validation White images:', len(os.listdir(validation_8_dir)))
print(' ')

print('total testing Black images:', len(os.listdir(test_0_dir)))
print('total testing Burgundy images:', len(os.listdir(test_1_dir)))
print('total testing Grey images:', len(os.listdir(test_2_dir)))
print('total testing Khaki images:', len(os.listdir(test_3_dir)))
print('total testing Orange images:', len(os.listdir(test_4_dir)))
print('total testing Red images:', len(os.listdir(test_5_dir)))
print('total testing Silver images:', len(os.listdir(test_6_dir)))
print('total testing Two tone images:', len(os.listdir(test_7_dir)))
print('total testing White images:', len(os.listdir(test_8_dir)))
print(' ')


# In[6]:


# Preparing zero-valued Numpy array for cut objects
# Shape: image number, height, width, number of channels
x_train = np.zeros((1, 150, 150, 3))

# Preparing temp zero-valued Numpy array for current cut object
# Shape: image number, height, width, number of channels
x_temp = np.zeros((1, 150, 150, 3))

# Defining boolean variable to track arrays' shapes
first_object = True

labels = ['Black', 'Burgundy', 'Grey', 'Khaki', 'Orange', 'Red', 'Silver', 'Two tone', 'White']
for label in labels:
    os.chdir(f'/content/drive/My Drive/Colab Notebooks/KP CNN/dataset warna/train/{label}')
    # Showing currently active directory
    print('Currently active directory:')
    print(os.getcwd())
    print()
    
    for current_dir, dirs, files in os.walk('.', topdown = True):
        # Iterating all files
        for f in files:
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

print(x_train.shape)
os.chdir('/content/drive/My Drive/Colab Notebooks/KP CNN')


# In[7]:


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

# In[22]:


train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1], featurewise_center=True)
val_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True)

train_datagen.fit(x_train)
val_datagen.fit(x_train)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=19,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='categorical')


# In[23]:


print(x_train.mean())
print(x_train.std())

print(train_datagen.mean)
print(train_datagen.std)


# # Eksperimen

# ## How Many ConvPool Pairs?

# ### Build Models to Evaluate 

# In[10]:


# Model ConvPool Pair (CPP) 1: RGB --> {8C5-P2} --> 128 --> 9
# Model ConvPool Pair (CPP) 2: RGB --> {8C5-P2} --> {16C5-P2} --> 128 --> 9
# Model ConvPool Pair (CPP) 3: RGB --> {8C5-P2} --> {16C5-P2} --> {32C5-P2} --> 128 --> 9
# Model ConvPool Pair (CPP) 4: RGB --> {8C5-P2} --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 128 --> 9

img_input = layers.Input(shape=(150, 150, 3))

# Building 1st CPP Model
x = layers.Conv2D(8, 5, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(9, activation='softmax')(x)

model_cpp_1 = Model(img_input, output)
model_cpp_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# Building 2nd CPP Model
x = layers.Conv2D(8, 5, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(9, activation='softmax')(x)

model_cpp_2 = Model(img_input, output)
model_cpp_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# Building 3rd CPP Model
x = layers.Conv2D(8, 5, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(9, activation='softmax')(x)

model_cpp_3 = Model(img_input, output)
model_cpp_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# Building 4th CPP Model
x = layers.Conv2D(8, 5, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(9, activation='softmax')(x)

model_cpp_4 = Model(img_input, output)
model_cpp_4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model_cpp = [model_cpp_1, model_cpp_2, model_cpp_3, model_cpp_4]


# ### Train Built Models 

# In[ ]:


print(" ")
print("TRAINING 1st Model")
print(" ")
history_cpp_1 = model_cpp_1.fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 2nd Model")
print(" ")
history_cpp_2 = model_cpp_2.fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 3rd Model")
print(" ")
history_cpp_3 = model_cpp_3.fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)
print(" ")
print("TRAINING 4th Model")
print(" ")
history_cpp_4 = model_cpp_4.fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)


# ### Evaluate (Visualization) 

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# ### Best Convolutional-Pooling Pairs: Model 3 (3 Pairs)
# 
# RGB --> {8C5-P2} --> {16C5-P2} --> {32C5-P2} --> 128 --> 9

# ## Feature Maps?

# ### Build Models to Evaluate 

# In[ ]:


# Model FM 1: RGB --> {8C5-P2} --> {16C5-P2} --> {32C5-P2}  --> 128 --> 9
# Model FM 2: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2}  --> 128 --> 9
# Model FM 3: RGB --> {32C5-P2} --> {64C5-P2} --> {128C5-P2}  --> 128 --> 9
# Model FM 4: RGB --> {64C5-P2} --> {128C5-P2} --> {256C5-P2}  --> 128 --> 9
# Model FM 5: RGB --> {128C5-P2} --> {256C5-P2} --> {512C5-P2}  --> 128 --> 9

# Defining list to collect models in
model_FM = []

# Building models in a loop
for i in range(5):
  x = layers.Conv2D(8*(2**i), 5, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(16*(2**i), 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(32*(2**i), 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  output = layers.Dense(9, activation='softmax')(x)

  temp = Model(img_input, output)
  temp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_FM.append(temp)


# ### Train Built Models 

# In[ ]:


# Defining list to collect results in
history_FM = []

# Training models in a loop
for i in range(5):
    print(" ")
    print(f"TRAINING Model FM {i+1}")
    print(" ")
    temp = model_FM[i].fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)
    history_FM.append(temp)


# ### Evaluate (Visualization) 

# In[ ]:


for i in range(5):
    print('Model FM {0}: MAX Training accuracy= {1:.5f}, MAX Validation accuracy= {2:.5f}, MAX Training loss= {3:.5f}, MAX Validation loss= {4:.5f}'.
                                                         format(i + 1,
                                                                max(history_FM[i].history['acc']),
                                                                max(history_FM[i].history['val_acc']),
                                                                max(history_FM[i].history['loss']),
                                                                max(history_FM[i].history['val_loss']),))


# In[ ]:


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


# In[ ]:


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
# RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 128 --> 9

# ## How Many Neurons in Output Layers?

# ### Build Models to Evaluate 

# In[11]:


# Model Neurons in Output Layers (NOL) 1: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 128 --> 9
# Model NOL 2: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 256 --> 9
# Model NOL 3: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 512 --> 9
# Model NOL 4: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 1024 --> 9
# Model NOL 5: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 2048 --> 9

model_nol = []

for i in range(5):
  x = layers.Conv2D(16, 5, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(32, 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(64, 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(128*(2**i), activation='relu')(x)
  output = layers.Dense(9, activation='softmax')(x)

  temp = Model(img_input, output)
  temp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_nol.append(temp)


# ### Train Built Models 

# In[12]:


# Defining list to collect results in
history_nol = []

# Training models in a loop
for i in range(5):
  print(" ")
  print(f"TRAINING Model NOL {i+1}")
  print(" ")
  temp = model_nol[i].fit(train_generator, steps_per_epoch=14, epochs=24, validation_data=validation_generator, validation_steps=4, verbose=1)
  history_nol.append(temp)


# ### Evaluate (Visualization) 

# In[13]:


for i in range(5):
    print('Model NOL {0}: MAX Training accuracy= {1:.5f}, MAX Validation accuracy= {2:.5f}, MAX Training loss= {3:.5f}, MAX Validation loss= {4:.5f}'.
          format(i + 1, max(history_nol[i].history['acc']), max(history_nol[i].history['val_acc']),
                 max(history_nol[i].history['loss']), max(history_nol[i].history['val_loss'])))


# In[14]:


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


# In[15]:


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


# ### Best number of neurons in output layer: 512 (Model 3)
# 
# RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 512 --> 9

# ## Best Preprocessing Method?

# ### Build Generator to Evaluate 

# In[16]:


# Generator 1: Pixel Scaling with normalization (1/255)
# Generator 2: Pixel scaling with centering (- mean image)
# Generator 3: Pixel scaling with standardizing (/ std image)

#Building generator 1
train_datagen_1 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1])
val_datagen_1 = ImageDataGenerator(rescale=1./255)

train_generator_1 = train_datagen_1.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=19,
        class_mode='categorical')

validation_generator_1 = val_datagen_1.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='categorical')



#Building generator 2
train_datagen_2 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1], featurewise_center=True)
val_datagen_2 = ImageDataGenerator(rescale=1./255, featurewise_center=True)

train_datagen_2.fit(x_train)
val_datagen_2.fit(x_train)

train_generator_2 = train_datagen_2.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=19,
        class_mode='categorical')

validation_generator_2 = val_datagen_2.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='categorical')



#Building generator 3
train_datagen_3 = ImageDataGenerator(rescale=1./255, zoom_range=0.1, horizontal_flip=True, brightness_range = [0.7, 1.1],
                                     featurewise_center=True, featurewise_std_normalization=True)
val_datagen_3 = ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True)

train_datagen_3.fit(x_train)
val_datagen_3.fit(x_train)

train_generator_3 = train_datagen_3.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=19,
        class_mode='categorical')

validation_generator_3 = val_datagen_3.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=14,
        class_mode='categorical')


# In[17]:


#Building model: RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 512 --> 9
model_gen = []
for i in range(3):
  x = layers.Conv2D(16, 5, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(32, 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(64, 5, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  output = layers.Dense(9, activation='softmax')(x)

  temp = Model(img_input, output)
  temp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  # Adding current model in the list
  model_gen.append(temp)


# ### Train Built Models 

# In[18]:


print(" ")
print(f"TRAINING WITH NORMALIZED DATA")
print(" ")
history_gen_1 = model_gen[0].fit(train_generator_1, steps_per_epoch=14, epochs=24, validation_data=validation_generator_1, validation_steps=4, verbose=1)

print(" ")
print(f"TRAINING WITH CENTRALIZED DATA")
print(" ")
history_gen_2 = model_gen[1].fit(train_generator_2, steps_per_epoch=14, epochs=24, validation_data=validation_generator_2, validation_steps=4, verbose=1)

print(" ")
print(f"TRAINING WITH STANDARDIZED DATA")
print(" ")
history_gen_3 = model_gen[2].fit(train_generator_3, steps_per_epoch=14, epochs=24, validation_data=validation_generator_3, validation_steps=4, verbose=1)


# ### Evaluate (Visualization) 

# In[19]:


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


# In[20]:


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


# In[21]:


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
# centralizing (feature-wise)

# ## Best Model Architecture:
# 
# RGB --> {16C5-P2} --> {32C5-P2} --> {64C5-P2} --> 512 --> 1
# 
# With feature-wised centralized data (x/255 - avg)

# # Data Modeling & Processing

# Because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

# Next, we'll configure the specifications for model training. We will train our model with the `binary_crossentropy` loss, because it's a binary classification problem and our final activation is a sigmoid. (For a refresher on loss metrics, see the [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture).) We will use the `adam` optimizer. During training, we will want to monitor classification accuracy.
# 
# **NOTE**: In this case, using the [RMSprop optimization algorithm](https://wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp) is preferable to [stochastic gradient descent](https://developers.google.com/machine-learning/glossary/#SGD) (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as [Adam](https://wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [Adagrad](https://developers.google.com/machine-learning/glossary/#AdaGrad), also automatically adapt the learning rate during training, and would work equally well here.)

# ## Building Model

# In[29]:


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 5x5, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 5, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 5x5, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 5x5, convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 5, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers  
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# # Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(9, activation='softmax')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully connected layer + sigmoid output layer
model = Model(img_input, output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model_warna.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the feature map.

# ## Training

# In[30]:


history = model.fit(
      train_generator,
      steps_per_epoch=14,  # 263 images = batch_size * steps
      epochs=50,
      validation_data=validation_generator,
      validation_steps=4,  # 52 images = batch_size * steps
      verbose=2, callbacks=[es, mc])


# ## Evaluating Accuracy and Loss for the Model
# 
# Let's plot the training/validation accuracy and loss as collected during training:

# In[31]:


# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc'][:16]
val_acc = history.history['val_acc'][:16]

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss'][:16]
val_loss = history.history['val_loss'][:16]

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
# plt.ylim([0, 2])
plt.title('Training and validation loss')

print('Best Model: Max Training accuracy= {0:.5f}, Max Validation accuracy= {1:.5f}, Min Training loss= {2:.5f}, Min Validation loss= {3:.5f}'
      .format(max(acc), max(val_acc),
              min(loss), min(val_loss)))


# ## Visualizing Intermediate Representations
# 
# To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.
# 
# Let's pick a random image from the training set, and then generate a figure where each row is the output of a layer, and each image in the row is a specific filter in that output feature map. Rerun this cell to generate intermediate representations for a variety of training images.

# In[32]:


# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
train_0_fnames = os.listdir(train_0_dir)
train_1_fnames = os.listdir(train_1_dir)
train_2_fnames = os.listdir(train_2_dir)
train_3_fnames = os.listdir(train_3_dir)
train_4_fnames = os.listdir(train_4_dir)
train_5_fnames = os.listdir(train_5_dir)
train_6_fnames = os.listdir(train_6_dir)
train_7_fnames = os.listdir(train_7_dir)
train_8_fnames = os.listdir(train_8_dir)

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random image from the training set.
black_img_files = [os.path.join(train_0_dir, f) for f in train_0_fnames]
burgundy_img_files = [os.path.join(train_1_dir, f) for f in train_1_fnames]
grey_img_files = [os.path.join(train_2_dir, f) for f in train_2_fnames]
khaki_img_files = [os.path.join(train_3_dir, f) for f in train_3_fnames]
orange_img_files = [os.path.join(train_4_dir, f) for f in train_4_fnames]
red_img_files = [os.path.join(train_5_dir, f) for f in train_5_fnames]
silver_img_files = [os.path.join(train_6_dir, f) for f in train_6_fnames]
twotone_img_files = [os.path.join(train_7_dir, f) for f in train_7_fnames]
white_img_files = [os.path.join(train_8_dir, f) for f in train_8_fnames]
img_path = random.choice(black_img_files + burgundy_img_files + grey_img_files + khaki_img_files + orange_img_files +
                        red_img_files + silver_img_files + twotone_img_files + white_img_files)

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

# ## Testing

# In[33]:


# load the saved model
saved_model = load_model('best_model_warna.h5')


# In[34]:


# load the test dataset
test_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True)
test_datagen.fit(x_train)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=1,
    shuffle = False,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

filenames = test_generator.filenames
nb_samples = len(filenames)


# In[35]:


#Predicting the test dataset using saved model
pred = model.predict(test_generator, steps = nb_samples)


# In[36]:


y_true = test_generator.classes
y_pred = []
for i in pred:
    y_pred.append(np.argmax(i))


# In[37]:


print(classification_report(y_true, y_pred))

c_m = confusion_matrix(y_true, y_pred)
print("Confusion Matrix: ")
print(c_m)


# In[38]:


labels = ['Black', 'Burgundy', 'Grey', 'Khaki', 'Orange', 'Red', 'Silver', 'Two tone', 'White']

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

