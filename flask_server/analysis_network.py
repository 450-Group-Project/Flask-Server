# imdb_lstm.py
# LSTM for sentiment analysis on the IMDB dataset

import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import load_model
import warnings

def analyze_text(text):
  # 0. get started
  print("IMDB sentiment analysis using Keras/TensorFlow")
  np.random.seed(1)
  tf.set_random_seed(1)

  # 1. load data into memory
  max_words = 20000
  print("Loading data, max unique words = %d words\n" % max_words)
  # save np.load
  np_load_old = np.load

  # modify the default parameters of np.load
  np.load = lambda *a, **k: np_load_old(*a, **k)

  # call load_data with allow_pickle implicitly set to true
  (train_x, train_y), (test_x, test_y) = \
      K.datasets.imdb.load_data(seed=1, num_words=max_words)

  # restore np.load for future normal usage
  np.load = np_load_old

  max_sentence_len = 80
  train_x = K.preprocessing.sequence.pad_sequences(train_x,
                                                   truncating='pre', padding='pre', maxlen=max_sentence_len)
  test_x = K.preprocessing.sequence.pad_sequences(test_x,
                                                  truncating='pre', padding='pre', maxlen=max_sentence_len)

  #predicts sentiment
  # 6. use model to make a prediction
  mp = ".\\Models\\imdb_model.h5"
  model = load_model(mp)
  print("New review: \'A contradictory statement is one that says two things that cannot both be true\'")
  d = K.datasets.imdb.get_word_index()
  word_or_sentence = text

  words = word_or_sentence.split()
  word_array = []
  for word in words:

      if word not in d:
          word_array.append(2)
      else:
          word_array.append(d[word] + 3)

  predata = K.preprocessing.sequence.pad_sequences([word_array],
                                                  truncating='pre', padding='pre', maxlen=max_sentence_len)
  prediction = model.predict(predata)
  print("Prediction (0 = negative, 1 = positive) = ", end="")
  ret = "%0.4f" % prediction[0][0]
  return ret


if __name__ == '__main__':
    main()