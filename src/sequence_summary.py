import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)  
print(data_dir)

# Data suite 1
train1_X = []
train1_Y = []
test1_X = []
test1_Y = []
validate1_X = []
validate1_Y = []

# Data suite 2
train2_X = []
train2_Y = []
test2_X = []
test2_Y = []
validate2_X = []
validate2_Y = []

# Data suite 2
train3_X = []
train3_Y = []
test3_X = []
test3_Y = []
validate3_X = []
validate3_Y = []

files = [('sequences100_test.csv',test1_X,test1_Y), ('sequences200_validate.csv',validate1_X,validate1_Y), ('sequences500_test.csv',train1_X,train1_Y), 
        ('sequences1000_train.csv',test2_X,test2_Y), ('sequences1000_validate.csv',validate2_X,validate2_Y), ('sequences5000_train.csv',train2_X,train2_Y),
        ('sequences10000_test.csv',test3_X,test3_Y), ('sequences20000_validate.csv',validate3_X,validate3_Y), ('sequences100000_train.csv',train3_X,train3_Y)]

for filename in files:
    file_path = os.path.join(data_dir, filename[0])
    print(file_path)
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            a = np.array([int(i) for i in row])
            # sequence of 64 numbers
            filename[1].append(a[:64])
            # 5 elements with greatest frequency then ascending
            filename[2].append(a[64:])

train3_X = np.array(train3_X)
train3_Y = np.array(train3_Y)
test3_X = np.array(test3_X)
test3_Y = np.array(test3_Y)
validate3_X = np.array(validate3_X)
validate3_Y = np.array(validate3_Y)
data_val_y = validate3_Y[:,:1]
data_y = train3_Y[:,:1]

input1 = Input(shape=(64,))
layer1 = Dense(40, activation='relu')(input1)
layer2 = Dense(120, activation='sigmoid')(layer1)
layer2 = Dense(60, activation='relu')(layer1)
outputs = Dense(20, activation='softmax')(layer2)

model = Model(inputs=input1, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

y_categorical = tf.keras.utils.to_categorical(data_y, num_classes=20)
y_categorical_val = tf.keras.utils.to_categorical(data_val_y, num_classes=20)

model.fit(train3_X, y_categorical, epochs=32, validation_data=(validate3_X, y_categorical_val), batch_size=8)

pred = model.predict(test3_X)
pred_labels = np.argmax(pred, axis=1) + 1  # Add 1 to shift from zero-based indexing to 1-20 range

# Print the predicted labels and the true labels
for i in range(100):
    print(f"Predicted: {pred_labels[i]}, True: {test2_Y[i]}")