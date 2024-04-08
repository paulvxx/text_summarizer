from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, TimeDistributed, TextVectorization, Input, LSTM, RepeatVector, Dense

#Limit testing data
test_limit = 500

# Define the TextVectorization layer
max_features = 20000  # Maximum vocabulary size
max_len = 1000         # Maximum length of a sequence
max_len_sum = 1      # Maximum summary text length
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len)

dataset = load_dataset("cnn_dailymail", '3.0.0')
train_data = dataset['train']
validation_data = dataset['validation']
first_tests = train_data[:test_limit]
first_valid = validation_data[:test_limit]

vectorize_layer.adapt(first_tests['article'])

text_src_data = []
text_sum_data = []

for i in range(test_limit):
    v = vectorize_layer(first_tests['highlights'][0])
    text_src_data.append(v.numpy())
    w = vectorize_layer(first_tests['highlights'][0])
    print(w[:1])
    text_sum_data.append(w[:1].numpy())

#v = vectorize_layer(first_tests['highlights'][0])
#print(v.numpy())
#vocab = vectorize_layer.get_vocabulary()
#print("Current vocabulary size:", len(vocab))

print(text_src_data[0])

# The model

#encoding

#max_len is 2000
text_vec = Input(shape=(max_len,))
embedding = Embedding(max_features, 128)(text_vec)
encoder1 = LSTM(128)(embedding)
# max summary length
encoder2 = RepeatVector(max_len_sum)(encoder1)
decoder1 = LSTM(128, return_sequences=True)(encoder2)
outputs = TimeDistributed(Dense(max_features, activation='softmax'))(decoder1)

# build the model
model = Model(inputs=text_vec, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
# additional Dense Layers
#decoding

print(text_src_data[0].shape)
print(text_sum_data[0].shape)

#print(np.array(text_sum_data).shape)

print(len(text_src_data)) # prints 1000 (1000 samples vectors of length 2000)
print(len(text_sum_data)) # prints 1000 (1000 samples of vectors of length 300)


tf_dataset = tf.data.Dataset.from_tensor_slices((text_src_data, text_sum_data)).batch(4)

model.fit(x = tf_dataset,
          epochs=4,
        )
