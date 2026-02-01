#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

def build_model1():
  # We define the input shape for CIFAR-10 explicitly
  input_shape = (32, 32, 3)
  model = tf.keras.Sequential([
  # 1. Input/Flatten
  tf.keras.layers.Flatten(input_shape=input_shape),
  # 2. Dense Layer 1 + LeakyReLU
  tf.keras.layers.Dense(128),
  tf.keras.layers.LeakyReLU(alpha=0.1),
  # 3. Dense Layer 2 + LeakyReLU
  tf.keras.layers.Dense(128),
  tf.keras.layers.LeakyReLU(alpha=0.1),

  # 4. Dense Layer 3 + LeakyReLU
  tf.keras.layers.Dense(128),
  tf.keras.layers.LeakyReLU(alpha=0.1),

  # 5. Output Layer (No activation, returns logits)
  tf.keras.layers.Dense(10) 
    ])
  # Compile
  model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  
  return model

def build_model2():
  input_shape = (32, 32, 3)    
  model = tf.keras.Sequential([
  # --- Input ---
  tf.keras.layers.Input(shape=input_shape),

  # --- Block 1: 32 filters, stride 2 ---
  tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),

  # --- Block 2: 64 filters, stride 2 ---
  tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),

  # --- Block 3: Four pairs of 128 filters (Stride 1) ---
  # Pair 1
  tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),
  # Pair 2
  tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),
  # Pair 3
  tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),
  # Pair 4
  tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
  tf.keras.layers.BatchNormalization(),

  # --- Output Block ---
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10) # Output logits
])

# Compile
  model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
  
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
 # --- Load and Prepare Data ---
  cifar10 = tf.keras.datasets.cifar10
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)

  # Choose num_val_samples indices up to the size of train_images, !replace => no repeats
  # We add a seed here just so your results are reproducible 
  np.random.seed(42) 
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)

  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]

  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]

  # 3. Squeeze Labels (Remove extra dimensions)
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  val_labels = val_labels.squeeze()

  # 4. Normalize (0 to 1)
  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0
  val_images   = val_images   / 255.0

  print("Training Images range from {:2.5f} to {:2.5f}".format(np.min(train_images), np.max(train_images)))
  print("Test     Images range from {:2.5f} to {:2.5f}".format(np.min(test_images), np.max(test_images)))
  print(f"Training set size: {train_images.shape}")
  print(f"Validation set size: {val_images.shape}")

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  print("\n--- Model 1 Summary ---")
  model1.summary()

  print("\nStarting training for Model 1...")
  # We save the training history to plot it later if needed
  history = model1.fit(train_images, train_labels, 
                         epochs=30, 
                         validation_data=(val_images, val_labels), verbose=2)
 
  print("\nEvaluating on Test Set...")
  test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=2)

  print(f"\nRESULTS:")
  print(f"Final Test Accuracy: {test_acc*100:.2f}%")


  ## Build, compile, and train model 2 (DS Convolutions)
  
  print("\nBuilding Model 2 (CNN)...")
  model2 = build_model2()
  model2.summary()

  print("\nStarting training for Model 2 (30 Epochs)...")
  history = model2.fit(train_images, train_labels, 
                        epochs=30, 
                        validation_data=(val_images, val_labels),
                        verbose=2)

  # --- 6. Evaluate on Test Set ---
  print("\nEvaluating Model 2 on Test Set...")
  test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=2)

  print(f"\nRESULTS for Model 2:")
  print(f"Final Test Accuracy: {test_acc*100:.2f}%")


  print("\n--- Testing on Real World Image ---")

  filename = 'test_image_cat.jpg' 

  try:
      # 1. Load the image using your specific syntax
      # We wrap it in np.array() exactly as requested
      test_img = np.array(tf.keras.utils.load_img(
            filename,
            grayscale=False,
            color_mode='rgb',
            target_size=(32,32)
      ))

      # 2. Preprocessing (Critical for correct results)
      # The model was trained on inputs 0-1, so we must normalize 0-255 inputs.
      test_img = test_img / 255.0
      
      # The model expects a batch of images (1, 32, 32, 3), not just one (32, 32, 3).
      test_img_input = np.expand_dims(test_img, axis=0)

      # 3. Predict
      # We use model2 (the CNN) as it is the best model so far
      predictions = model2.predict(test_img_input)
      predicted_class_idx = np.argmax(predictions)
      predicted_label = class_names[predicted_class_idx]

      # 4. Output Results
      print(f"Loaded file: {filename}")
      print(f"Model prediction: {predicted_label.upper()}")
      
      # Helper check
      if predicted_label in filename:
          print("Does it correctly label the picture? YES")
      else:
          print("Does it correctly label the picture? NO (Check manually)")

  except Exception as e:
      print(f"Error loading image: {e}")
      print(f"Make sure '{filename}' is in the same folder as this script.")

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
