from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Embedding, TimeDistributed, TextVectorization, Input, LSTM, RepeatVector, Dense, concatenate

#Limit testing data
test_limit = 200
test_val_limit = round(test_limit/5)

# Define the TextVectorization layer
max_features = 20000  # Maximum vocabulary size
max_len = 1000         # Maximum length of a sequence
max_len_sum = 100      # Maximum summary text length
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len)

dataset = load_dataset("cnn_dailymail", '3.0.0')
train_data = dataset['train']
validation_data = dataset['validation']
first_tests = train_data[:test_limit]
first_valid = validation_data[:test_val_limit]

vectorize_layer.adapt(first_tests['article'])

text_src_data = []
text_sum_data = []
text_src_val_data = []
text_sum_val_data = []

for i in range(test_limit):
    v = vectorize_layer(first_tests['article'][i])
    text_src_data.append(v.numpy())
    w = vectorize_layer(first_tests['highlights'][i])
    text_sum_data.append(w[:max_len_sum].numpy())

for i in range(test_val_limit):
    v = vectorize_layer(first_valid['article'][i])
    text_src_val_data.append(v.numpy())
    w = vectorize_layer(first_valid['highlights'][i])
    text_sum_val_data.append(w[:max_len_sum].numpy())

#v = vectorize_layer(first_tests['highlights'][0])
#print(v.numpy())
#vocab = vectorize_layer.get_vocabulary()
#print("Current vocabulary size:", len(vocab))

print(text_src_data[0])

# The model

#encoding
# Source text input
input1 = Input(shape=(max_len,))
embedding = Embedding(max_features, 128)(input1)
layer1 = LSTM(128)(embedding)
layer2 = RepeatVector(max_len_sum)(layer1)

# Partial summary input (current summary --- all previously guessed words)
input2 = Input(shape=(max_len_sum,))
layer3 = Embedding(max_features, 128)(input2)

# Decoder output (next word)
decoder1 = concatenate([layer2, layer3])
decoder2 = LSTM(128)(decoder1)
outputs = Dense(max_features, activation='softmax')(decoder2)

# build the model
optimizer = Adam(learning_rate=0.02)

model = Model(inputs=[input1, input2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# additional Dense Layers
#decoding

print(text_src_data[0].shape)
print(text_sum_data[0].shape)

text_src_data = np.array(text_src_data)
text_sum_data = np.array(text_sum_data)
text_src_val_data = np.array(text_src_val_data)
text_sum_val_data = np.array(text_sum_val_data)

#initialize array of zeroes to represent subsequent word predictions
text_prediction_data = np.zeros((test_limit, max_len_sum), dtype=int)
text_prediction_val_data = np.zeros((test_val_limit, max_len_sum), dtype=int)

print(text_src_data.shape)
print(text_sum_data.shape)
print(text_prediction_data)

#print(len(text_src_data)) # prints 1000 (1000 samples vectors of length 2000)
#print(len(text_sum_data)) # prints 1000 (1000 samples of vectors of length 300)


for i in range(10):        
    #tf_dataset = tf.data.Dataset.from_tensor_slices((text_src_data, text_sum_data)).batch(4)
    #print(text_sum_data[0])

    model.fit([text_src_data, text_prediction_data],
            text_sum_data[:,i:i+1],
            epochs=4,
            batch_size=8, 
            validation_data=([text_src_val_data, text_prediction_val_data], text_sum_val_data[:,i:i+1]),
            )

    pred = model.predict([text_src_data, text_prediction_data])
    pred_val = model.predict([text_src_data, text_prediction_data])

    for j in range(test_limit):
        # get argmax (highest probability word) for next word in summary
        text_prediction_data[j][i] = np.argmax(pred[j])

    for j in range(test_val_limit):
        text_prediction_val_data[j][i] = np.argmax(pred_val[j])

    print(text_prediction_data[5])
    print(text_sum_data[5])

    print(text_prediction_data[17])
    print(text_sum_data[17])
