'''
Trains model used to generate answers

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''

import io
import os
import json
import re
import random
import sys
import numpy as np
from attention_model import AttentionGRUModel
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding, Conv1D, Masking, Flatten

GLOVE_DIR = '../../data'
CONTEXTS_PATH = '../../data/contexts.test'
ANSWERS_PATH = '../../data/answers.test'
QUESTIONS_PATH = '../../data/questions.test'

C_TOK_PATH = '../../data/context_tok.json'
A_TOK_PATH = '../../data/answer_tok.json'
Q_TOK_PATH = '../../data/question_tok.json'
MODEL_PATH = '../../data/model.h5'

np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1337)
vocab_size = 10000

def preprocess(line, entire = False):
  ltext = line.replace('\n', ' ').strip()
  if entire:      
    ltext = re.sub('[^0-9a-zA-Z]+', ' ', ltext)
    ltext = ltext.lower()
  else:
    ltext = ltext.replace('\n', ' ')
    ltext = ltext.replace(',', '')
    ltext = ltext.replace(':', '')
    ltext = ltext.replace("'", '')
  return ltext

def train_main():
  # Preprocess / Split
  dat = []
  with open(CONTEXTS_PATH, 'r')  as c_p, \
      open(ANSWERS_PATH, 'r')   as a_p, \
      open(QUESTIONS_PATH, 'r') as q_p:
    for question_line in q_p.readlines():
      try:
        context = preprocess(c_p.readline(), False)
        answer = preprocess(a_p.readline(), False)
        question = preprocess(question_line, True)
        if context and answer and question:
          dat.append((context, answer, question))
      except ValueError as v:
        pass # line was blank

  # Split
  random.shuffle(dat)
  context, answer, question = zip(*dat)

  trainlen = int(len(answer) * 0.85)
  vallen = int(len(answer) * 0.05)
  testlen = int(len(answer) * 0.10)
  
  trainquestion = question[:trainlen]
  valquestion = question[trainlen:trainlen+vallen]
  testquestion = question[trainlen+vallen:]

  traincontext = context[:trainlen]
  valcontext = context[trainlen:trainlen+vallen]
  testcontext = context[trainlen+vallen:]

  trainanswer = answer[:trainlen]
  valanswer = answer[trainlen:trainlen+vallen]
  testanswer = answer[trainlen+vallen:]

  # Set up tokenizers
  question_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  question_tokenizer.fit_on_texts(trainquestion)

  answer_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  answer_tokenizer.fit_on_texts(trainanswer)

  context_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  context_tokenizer.fit_on_texts(traincontext)

  q_tokenizer_json = question_tokenizer.to_json()
  a_tokenizer_json = answer_tokenizer.to_json()
  c_tokenizer_json = context_tokenizer.to_json()

  with open(Q_TOK_PATH, 'w') as f:
    f.write(q_tokenizer_json)
  with open(A_TOK_PATH, 'w') as f:
    f.write(a_tokenizer_json)
  with open(C_TOK_PATH, 'w') as f:
    f.write(c_tokenizer_json)

  # Using GLOVE word embeddings
  # Source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
  embeddings_index = {}
  vector_size = 100
  word_index = question_tokenizer.word_index
  with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

  embedding_matrix = np.zeros((vocab_size, vector_size))
  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all-zeros.
      embedding_matrix[i] = embedding_vector

  # Tokenize and pad everything
  # set up config
  config = dict()
  config['q_vocabsize'] = vocab_size
  config['c_vocabsize'] = vocab_size
  config['a_vocabsize'] = vocab_size
  config['qlen'] = 10
  config['clen'] = 1000
  config['alen'] = 10 # answers won't be very long
  batch_size = 100
  config['batch_size'] = batch_size
  config['weights'] = [embedding_matrix]

  trainquestion = question_tokenizer.texts_to_sequences(trainquestion)
  valquestion = question_tokenizer.texts_to_sequences(valquestion)
  testquestion = question_tokenizer.texts_to_sequences(testquestion)

  trainanswer = answer_tokenizer.texts_to_sequences(trainanswer)
  valanswer = answer_tokenizer.texts_to_sequences(valanswer)
  testanswer = answer_tokenizer.texts_to_sequences(testanswer)

  traincontext = context_tokenizer.texts_to_sequences(traincontext)
  valcontext = context_tokenizer.texts_to_sequences(valcontext)
  testcontext = context_tokenizer.texts_to_sequences(testcontext)

  trainquestion = pad_sequences(trainquestion, padding="post", truncating="post", maxlen=config['qlen'])
  valquestion = pad_sequences(valquestion, padding="post", truncating="post", maxlen=config['qlen'])
  testquestion = pad_sequences(testquestion, padding="post", truncating="post", maxlen=config['qlen'])

  trainanswer = pad_sequences(trainanswer, padding="post", truncating="post", maxlen=config['alen'])
  valanswer = pad_sequences(valanswer, padding="post", truncating="post", maxlen=config['alen'])
  testanswer = pad_sequences(testanswer, padding="post", truncating="post", maxlen=config['alen'])

  traincontext = pad_sequences(traincontext, padding="post", truncating="post", maxlen=config['clen'])
  valcontext = pad_sequences(valcontext, padding="post", truncating="post", maxlen=config['clen'])
  testcontext = pad_sequences(testcontext, padding="post", truncating="post", maxlen=config['clen'])

  # Train

  # create model
  mdl = AttentionGRUModel(config)
  model = mdl.create_model()
  print(model.summary())

  K.set_value(model.optimizer.learning_rate, 0.001)

  history = model.fit([trainquestion, trainanswer, traincontext], trainanswer,
                      batch_size=batch_size,
                      epochs=5,
                      verbose=1,
                      validation_data=([valquestion, valanswer, valcontext], valanswer))

  # Save model
  model.save(MODEL_PATH)
  Ypred = model.predict((testcontext, testquestion, testanswer))
  Ypred = np.argmax(Ypred, axis=1)
  Ytest = np.argmax(Ytest, axis=1)

  print(metrics.classification_report(Ytest, Ypred))

if __name__ == '__main__':
  train_main()