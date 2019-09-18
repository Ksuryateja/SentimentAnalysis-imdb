from keras.datasets import 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)
print('Number of Training samples: {}\n Number of Test samples:{}'.format(len(X_train), len(X_test)))

# Pad the sequences to a fixed length
max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen = max_words)
X_test = sequence.pad_sequences(X_test, maxlen = max_words)

# Model
embedding_size = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = max_words))
model.add(LSTM(100))
model.add(Dropout(rate = 0.25))
model.add(Dense(1, activation = 'sigmoid'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Test and Val split
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

filepath="SA-{epoch:02d}-{val_acc:.2f}"
checkpoint = ModelCheckpoint("./models/{}.hdf5".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

model.fit(x = X_train, y = y_train, batch_size = 128, epochs = 10, validation_data = (X_val, y_val), callbacks = [checkpoint])

history = model.history.history

def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

plot_metrics(history)

model.load_weights('./models/SA-06-0.88.hdf5')
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss: {}\nAccuracy: {}'.format(loss, accuracy))