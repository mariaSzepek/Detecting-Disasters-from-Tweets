import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn import model_selection, metrics, preprocessing, ensemble, model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from google.colab import files

data = pd.read_csv("Data Mining.csv")

raw_data = pd.read_csv("Data Mining.csv")
print("Data points count: ", raw_data['id'].count())
raw_data.head()

# Plotting target value counts
plt.figure(figsize=(10, 8))
ax = raw_data['target'].value_counts().sort_values().plot(kind="bar")
ax.grid(axis="y")
plt.suptitle("Target Value Counts", fontsize=20)
plt.show()

print("Number of missing data for column keyword: ", raw_data['keyword'].isna().sum())
print("Number of missing data for column location: ", raw_data['location'].isna().sum())
print("Number of missing data for column text: ", raw_data['text'].isna().sum())
print("Number of missing data for column target: ", raw_data['target'].isna().sum())

plt.figure(figsize=(15, 8))
sns.heatmap(raw_data.drop('id', axis=1).isnull(), cbar=False, cmap="GnBu").set_title("Missing data for each column")
plt.show()

plt.figure(figsize=(15, 8))
raw_data['word_count'] = raw_data['text'].apply(lambda x: len(x.split(" ")))
sns.distplot(raw_data['word_count'].values, hist=True, kde=True, kde_kws={"shade": True})
plt.axvline(raw_data['word_count'].describe()['25%'], ls="--")
plt.axvline(raw_data['word_count'].describe()['50%'], ls="--")
plt.axvline(raw_data['word_count'].describe()['75%'], ls="--")

plt.grid()
plt.suptitle("Word count histogram")
plt.show()

# remove rows with under 3 words
raw_data = raw_data[raw_data['word_count'] > 2]
raw_data = raw_data.reset_index()

print("25th percentile: ", raw_data['word_count'].describe()['25%'])
print("mean: ", raw_data['word_count'].describe()['50%'])
print("75th percentile: ", raw_data['word_count'].describe()['75%'])

import nltk

nltk.download('punkt')

# Clean text columns
# stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(each_text):
    # remove URL from text
    each_text_no_url = re.sub(r"http\S+", "", each_text)

    # remove numbers from text
    text_no_num = re.sub(r'\d+', '', each_text_no_url)

    # tokenize each text
    word_tokens = word_tokenize(text_no_num)

    # remove sptial character
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))

    # remove  lower
    text_with_no_stop_word = [w.lower() for w in clean_text]  # if not w in stop_words

    # do stemming
    stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]

    return " ".join(" ".join(stemmed_text).split())


raw_data['clean_text'] = raw_data['text'].apply(lambda x: clean_text(x))
raw_data['keyword'] = raw_data['keyword'].fillna("none")
raw_data['clean_keyword'] = raw_data['keyword'].apply(lambda x: clean_text(x))

# Combine column 'clean_keyword' and 'clean_text' into one
raw_data['keyword_text'] = raw_data['clean_keyword'] + " " + raw_data["clean_text"]

feature = 'keyword_text'
label = "target"

# split train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(raw_data[feature],
                                                                    raw_data[label],
                                                                    test_size=0.3,
                                                                    random_state=0,
                                                                    shuffle=True)

X_train

# Vectorize text
vectorizer = CountVectorizer()


# url = "http://nlp.stanford.edu/data/glove.6B.zip"
# Get the data with wget and the URL

!wget - cq
http: // nlp.stanford.edu / data / glove
.6
B.zip

# now you have the compressed data uploaded to your Colab, unzip it using the following command line:
import zipfile

zip_ref = zipfile.ZipFile("/content/glove.6B.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Define parameters
path_to_glove_file = '/content/glove.6B.300d.txt'  # download link: http://nlp.stanford.edu/data/glove.6B.zip
embedding_dim = 300
learning_rate = 1e-3
batch_size = 1024
epochs = 20
sequence_len = 100

y_train

# Define train and test labels
y_train_LSTM = y_train.values.reshape(-1, 1)
y_test_LSTM = y_test.values.reshape(-1, 1)

print("Training Y shape:", y_train_LSTM.shape)
print("Testing Y shape:", y_test_LSTM.shape)

y_train_LSTM

# Tokenize train data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocabulary Size: ", vocab_size)

# Pad train and test
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=sequence_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=sequence_len)

print("Training X shape: ", X_train.shape)
print("Testing X shape: ", X_test.shape)

X_train

# Read word embeddings
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

embeddings_index

# Define embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
embedding_matrix

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Activation

# Build and compile the model
model = Sequential()

# Build and compile the model 2: 2 hidden layers


embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            embedding_dim,
                                            weights=[embedding_matrix],
                                            input_length=sequence_len,
                                            trainable=False)
model2 = Sequential()  # True

model2.add(embedding_layer)

model2.add(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))
model2.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.2))  ##evtl behalten

model2.add(Dropout(rate=0.5))
model2.add(Dense(1, activation='sigmoid'))
model2.summary()

# This is for metrics cause its been thrown out see: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


import tensorflow as tf

# model.compile( ..., metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall()])
# , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
# 'accuracy'

embedding_dim = 300
learning_rate = 1e-3
batch_size = 1024
epochs = 20
sequence_len = 100

# compile the model
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
# Adam(learning_rate=learning_rate)
# 'adam'


# Train the LSTM Model
history = model2.fit(X_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(X_test, y_test))

# Evaluate the model
predicted = model2.predict(X_test, verbose=1, batch_size=10000)

y_predicted = [1 if each > 0.5 else 0 for each in predicted]

score, test_accuracy, a, b, c = model2.evaluate(X_test, y_test, batch_size=10000)

print("Test Accuracy: ", test_accuracy)
print(metrics.classification_report(list(y_test), y_predicted))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model2.evaluate(X_test, y_test, verbose=0)

# Print the obtained loss and accuracy
# print("Loss: {0}\nAccuracy: {1}\nAccuracy: {2}\nAccuracy: {3}".format(*model.evaluate(X_test, y_test, verbose=0)))
print("Loss: {0}\nAccuracy: {1}\nF1-Score: {2}\nPrecision: {3}\nRecall: {4}".format(
    *model2.evaluate(X_test, y_test, verbose=0)))

# Compute the confusion matrix- False positives, False négatives, True positives, True
# negatives

# Plot confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, cbar=False, cmap='Reds', annot=True, fmt='d')
ax.set(xlabel="Predicted Value", ylabel="True Value", title="Confusion Matrix")
ax.set_yticklabels(labels=['0', '1'], rotation=0)
plt.show()

# Plot the accuracy and loss (y-axis) and epochs (x-axis)

# Plot train accuracy and loss
accuraties = history.history['accuracy']
losses = history.history['loss']
accuraties_losses = list(zip(accuraties, losses))

accuraties_losses_df = pd.DataFrame(accuraties_losses, columns={"accuraties", "losses"})

plt.figure(figsize=(10, 4))
plt.suptitle("Train Accuracy vs Train Loss")
sns.lineplot(data=accuraties_losses_df)
plt.show()

# Model 1

# Build and compile the model 1 : 1 hidden layer
model = Sequential()

model.add(embedding_layer)

model.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.2))  ##evtl behalten

model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])

# Train the LSTM Model
history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))

# Evaluate the model
predicted = model.predict(X_test, verbose=1, batch_size=10000)

y_predicted = [1 if each > 0.5 else 0 for each in predicted]

score, test_accuracy, a, b, c = model.evaluate(X_test, y_test, batch_size=10000)

print("Test Accuracy: ", test_accuracy)
print(metrics.classification_report(list(y_test), y_predicted))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model2.evaluate(X_test, y_test, verbose=0)

# Print the obtained loss and accuracy
# print("Loss: {0}\nAccuracy: {1}\nAccuracy: {2}\nAccuracy: {3}".format(*model.evaluate(X_test, y_test, verbose=0)))
print("Loss: {0}\nAccuracy: {1}\nF1-Score: {2}\nPrecision: {3}\nRecall: {4}".format(
    *model2.evaluatimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn import model_selection, metrics, preprocessing, ensemble, model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from google.colab import files



data= pd.read_csv("Data Mining.csv")

raw_data = pd.read_csv("Data Mining.csv")
print("Data points count: ", raw_data['id'].count())
raw_data.head()

# Plotting target value counts
plt.figure(figsize=(10,8))
ax = raw_data['target'].value_counts().sort_values().plot(kind="bar")
ax.grid(axis="y")
plt.suptitle("Target Value Counts", fontsize=20)
plt.show()

print("Number of missing data for column keyword: ", raw_data['keyword'].isna().sum())
print("Number of missing data for column location: ", raw_data['location'].isna().sum())
print("Number of missing data for column text: ", raw_data['text'].isna().sum())
print("Number of missing data for column target: ", raw_data['target'].isna().sum())

plt.figure(figsize=(15,8))
sns.heatmap(raw_data.drop('id', axis=1).isnull(), cbar=False, cmap="GnBu").set_title("Missing data for each column")
plt.show()

plt.figure(figsize=(15,8))
raw_data['word_count'] = raw_data['text'].apply(lambda x: len(x.split(" ")) )
sns.distplot(raw_data['word_count'].values, hist=True, kde=True, kde_kws={"shade": True})
plt.axvline(raw_data['word_count'].describe()['25%'], ls="--")
plt.axvline(raw_data['word_count'].describe()['50%'], ls="--")
plt.axvline(raw_data['word_count'].describe()['75%'], ls="--")

plt.grid()
plt.suptitle("Word count histogram")
plt.show()

# remove rows with under 3 words
raw_data = raw_data[raw_data['word_count']>2]
raw_data = raw_data.reset_index()

print("25th percentile: ", raw_data['word_count'].describe()['25%'])
print("mean: ", raw_data['word_count'].describe()['50%'])
print("75th percentile: ", raw_data['word_count'].describe()['75%'])

import nltk
nltk.download('punkt')

# Clean text columns
#stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(each_text):

    # remove URL from text
    each_text_no_url = re.sub(r"http\S+", "", each_text)

    # remove numbers from text
    text_no_num = re.sub(r'\d+', '', each_text_no_url)

    # tokenize each text
    word_tokens = word_tokenize(text_no_num)

    # remove sptial character
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))

    # remove  lower
    text_with_no_stop_word = [w.lower() for w in clean_text] # if not w in stop_words

    # do stemming
    stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]

    return " ".join(" ".join(stemmed_text).split())


raw_data['clean_text'] = raw_data['text'].apply(lambda x: clean_text(x) )
raw_data['keyword'] = raw_data['keyword'].fillna("none")
raw_data['clean_keyword'] = raw_data['keyword'].apply(lambda x: clean_text(x) )

# Combine column 'clean_keyword' and 'clean_text' into one
raw_data['keyword_text'] = raw_data['clean_keyword'] + " " + raw_data["clean_text"]

feature = 'keyword_text'
label = "target"

# split train and test
X_train, X_test,y_train, y_test = model_selection.train_test_split(raw_data[feature],
                                                                   raw_data[label],
                                                                   test_size=0.3,
                                                                   random_state=0,
                                                                   shuffle=True)

X_train

# Vectorize text
vectorizer = CountVectorizer()


# url = "http://nlp.stanford.edu/data/glove.6B.zip"
# Get the data with wget and the URL

!wget -cq http://nlp.stanford.edu/data/glove.6B.zip


#now you have the compressed data uploaded to your Colab, unzip it using the following command line:
import zipfile

zip_ref = zipfile.ZipFile("/content/glove.6B.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Define parameters
path_to_glove_file = '/content/glove.6B.300d.txt' # download link: http://nlp.stanford.edu/data/glove.6B.zip
embedding_dim = 300
learning_rate = 1e-3
batch_size = 1024
epochs = 20
sequence_len = 100



y_train

# Define train and test labels
y_train_LSTM = y_train.values.reshape(-1,1)
y_test_LSTM = y_test.values.reshape(-1,1)

print("Training Y shape:", y_train_LSTM.shape)
print("Testing Y shape:", y_test_LSTM.shape)

y_train_LSTM

# Tokenize train data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocabulary Size: ", vocab_size)

# Pad train and test
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=sequence_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=sequence_len)

print("Training X shape: ", X_train.shape)
print("Testing X shape: ", X_test.shape)

X_train

# Read word embeddings
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

embeddings_index

# Define embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
embedding_matrix


embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Activation

# Build and compile the model
model = Sequential()

# Build and compile the model 2: 2 hidden layers


embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            embedding_dim,
                                            weights=[embedding_matrix],
                                            input_length=sequence_len,
                                            trainable=False)
model2 = Sequential() # True


model2.add(embedding_layer)

model2.add(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))
model2.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.2)) ##evtl behalten

model2.add(Dropout(rate=0.5))
model2.add(Dense(1, activation='sigmoid'))
model2.summary()



# This is for metrics cause its been thrown out see: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


import tensorflow as tf

# model.compile( ..., metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall()])
# , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
# 'accuracy'

embedding_dim = 300
learning_rate = 1e-3
batch_size = 1024
epochs = 20
sequence_len = 100

# compile the model
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
#Adam(learning_rate=learning_rate)
# 'adam'


# Train the LSTM Model
history = model2.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))







# Evaluate the model
predicted = model2.predict(X_test, verbose=1, batch_size=10000)

y_predicted = [1 if each > 0.5 else 0 for each in predicted]

score, test_accuracy, a, b, c = model2.evaluate(X_test, y_test, batch_size=10000)

print("Test Accuracy: ", test_accuracy)
print(metrics.classification_report(list(y_test), y_predicted))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model2.evaluate(X_test, y_test, verbose=0)

# Print the obtained loss and accuracy
#print("Loss: {0}\nAccuracy: {1}\nAccuracy: {2}\nAccuracy: {3}".format(*model.evaluate(X_test, y_test, verbose=0)))
print("Loss: {0}\nAccuracy: {1}\nF1-Score: {2}\nPrecision: {3}\nRecall: {4}".format(*model2.evaluate(X_test, y_test, verbose=0)))


# Compute the confusion matrix- False positives, False négatives, True positives, True
# negatives

# Plot confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, cbar=False, cmap='Reds', annot=True, fmt='d')
ax.set(xlabel="Predicted Value", ylabel="True Value", title="Confusion Matrix")
ax.set_yticklabels(labels=['0', '1'], rotation=0)
plt.show()

# Plot the accuracy and loss (y-axis) and epochs (x-axis)

# Plot train accuracy and loss
accuraties = history.history['accuracy']
losses = history.history['loss']
accuraties_losses = list(zip(accuraties,losses))

accuraties_losses_df = pd.DataFrame(accuraties_losses, columns={"accuraties", "losses"})

plt.figure(figsize=(10,4))
plt.suptitle("Train Accuracy vs Train Loss")
sns.lineplot(data=accuraties_losses_df)
plt.show()



# Model 1

# Build and compile the model 1 : 1 hidden layer
model = Sequential()



model.add(embedding_layer)

model.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.2)) ##evtl behalten

model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])

# Train the LSTM Model
history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))

# Evaluate the model
predicted = model.predict(X_test, verbose=1, batch_size=10000)

y_predicted = [1 if each > 0.5 else 0 for each in predicted]

score, test_accuracy, a, b, c = model.evaluate(X_test, y_test, batch_size=10000)

print("Test Accuracy: ", test_accuracy)
print(metrics.classification_report(list(y_test), y_predicted))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model2.evaluate(X_test, y_test, verbose=0)

# Print the obtained loss and accuracy
#print("Loss: {0}\nAccuracy: {1}\nAccuracy: {2}\nAccuracy: {3}".format(*model.evaluate(X_test, y_test, verbose=0)))
print("Loss: {0}\nAccuracy: {1}\nF1-Score: {2}\nPrecision: {3}\nRecall: {4}".format(*model2.evaluate(X_test, y_test, verbose=0)))

# Plot confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, cbar=False, cmap='Reds', annot=True, fmt='d')
ax.set(xlabel="Predicted Value", ylabel="True Value", title="Confusion Matrix")
ax.set_yticklabels(labels=['0', '1'], rotation=0)
plt.show()

# Plot train accuracy and loss
accuraties = history.history['accuracy']
losses = history.history['loss']
accuraties_losses = list(zip(accuraties,losses))

accuraties_losses_df = pd.DataFrame(accuraties_losses, columns={"accuraties", "losses"})

plt.figure(figsize=(10,4))
plt.suptitle("Train Accuracy vs Train Loss")
sns.lineplot(data=accuraties_losses_df)
plt.show()