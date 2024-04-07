from datasets import load_dataset
from tensorflow.keras.layers import Embedding, TextVectorization

# Define the TextVectorization layer
max_features = 100000  # Maximum vocabulary size
max_len = 500         # Maximum length of a sequence
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len)


dataset = load_dataset("cnn_dailymail", '3.0.0')
train_data = dataset['train']

first_five = train_data['article'][:5]

#print(len(first_five))
#print(first_five[0][:15])
#print(first_five[1][:15])
#print(first_five[2][:15])
#print(first_five[3][:15])
#print(first_five[4][:15])

vectorize_layer.adapt(first_five)

print(train_data['highlights'][0])

v = vectorize_layer(train_data['highlights'][0])

print(v.numpy())

vocab = vectorize_layer.get_vocabulary()
print("Current vocabulary size:", len(vocab))

#print(train_data[0]['article'])
#print(train_data[0]['highlights'])
#print(train_data[0]['id'])
