# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:18:20 2021

@author: jasmi
"""
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import to_categorical

df = pd.read_csv (r'C:\Users\jasmi\Downloads\heart\heart_failure_clinical_records_dataset.csv')

train_set = df[0:math.ceil(len(df)*0.7)]
test_set  = df[math.ceil(len(df)*0.7)+1:]

train_labels = to_categorical(train_set.DEATH_EVENT)
test_labels = to_categorical(test_set.DEATH_EVENT)

train_set = train_set[train_set.columns[:-1]]
test_set = test_set[test_set.columns[:-1]]

#train_set=(train_set-train_set.min())/(train_set.max()-train_set.min())
#test_set=(test_set-test_set.min())/(test_set.max()-test_set.min())

train_set = tf.convert_to_tensor(train_set, dtype=tf.int64) 
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64) 

network = models.Sequential()
network.add(layers.Dense(50,activation='relu', input_shape=(12,)))
network.add(layers.Dense(20,activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
network.summary()

opt = optimizers.Adam(learning_rate=0.01)
network.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(train_set, 
                      train_labels, 
                      epochs=20, 
                      batch_size=128, 
                      validation_data=(test_set,test_labels))

test_loss, test_acc = network.evaluate(test_set, test_labels)
print('test_acc: ', test_acc)


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len( history_dict['loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
