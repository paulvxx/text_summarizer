from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", '3.0.0')
train_data = dataset['train']

#print(train_data[0]['article'])
#print(train_data[0]['highlights'])
#print(train_data[0]['id'])
