import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.chdir(os.getcwd())
df = pd.read_csv('imdb_labelled.txt', sep='\t', header=None)
print(df.shape)
df.columns = ['text', 'sentiment']
print(df.head())

text = df.text.values
labels = df.sentiment.values

num_tokens = 2000
max_len = 20

tokenize = Tokenizer(num_words=num_tokens, oov_token='<OOV>')
tokenize.fit_on_texts(text)
sentences = tokenize.texts_to_sequences(text)

text_padded = pad_sequences(sentences, maxlen=max_len, padding='post', truncating='post')

print(text_padded)
print(text_padded.shape)

X_train, X_test, y_train, y_test = train_test_split(text_padded, labels, test_size=0.25, random_state=41)

initializer1 = tf.keras.initializers.Orthogonal(seed=35)
initializer2 = tf.keras.initializers.Orthogonal(seed=37)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_tokens, 16, input_length=max_len, embeddings_initializer=initializer1),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer2)
])
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, epochs=30, verbose=1)

y_pred = np.round(model.predict(X_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
