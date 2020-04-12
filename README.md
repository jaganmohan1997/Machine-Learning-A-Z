# Machine-Learning-A-Z
Exploring Machine Learning from A to Z

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



#Initilizating CNN

classification = Sequential()



# Step 1 -> Convolution

#classification.add(Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu'))

classification.add(Conv2D(32,(3,3),input_shape=(32,32,1),activation='relu'))



# Step 2 -> Max Pooling

classification.add(MaxPooling2D(pool_size=(2,2)))



# Step 3 -> Flattening

classification.add(Flatten())



# Step 4 -> Full Connection

classification.add(Dense(units=128,activation='relu'))



# Step 5 -> Combining all the steps together to create the CNN

classification.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('dataset/train_data',

                                                 target_size = (32, 32),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('dataset/test_data',

                                            target_size = (32, 32),

                                            batch_size = 32,

                                            class_mode = 'binary')



classification.fit_generator(training_set,

                         steps_per_epoch = 4000,

                         epochs = 15,

                         validation_data = test_set,

                         validation_steps = 1000)
