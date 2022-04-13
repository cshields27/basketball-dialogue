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

np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1337)
vocab_size = 10000

def preprocess(line, entire = False):
  ltext = line.replace('\n', ' ').strip()
  if entire:      
    ltext = re.sub('[^0-9a-zA-Z]+', ' ', ltext)        
    ltext = ltext.lower()
  return ltext

def train_main():
  # Preprocess / Split
  dat = []
  with open(CONTEXTS_PATH, 'r')  as c_p, \
      open(ANSWERS_PATH, 'r')   as a_p, \
      open(QUESTIONS_PATH, 'r') as q_p:
    for question_line in q_p.readlines():
      context = preprocess(c_p.readline(), False)
      answer = preprocess(a_p.readline(), False)
      try:
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

  traincontext = context[:trainlen]
  valcontext = context[trainlen:trainlen+vallen]
  testcontext = context[trainlen+vallen:]

  trainanswer = answer[:trainlen]
  valanswer = answer[trainlen:trainlen+vallen]
  testanswer = answer[trainlen+vallen:]

  trainquestion = question[:trainlen]
  valquestion = question[trainlen:trainlen+vallen]
  testquestion = question[trainlen+vallen:]

  # Tokenize
  question_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  question_tokenizer.fit_on_texts(trainquestion)

  answer_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  answer_tokenizer.fit_on_texts(trainanswer)

  context_tokenizer = Tokenizer(lower=False, num_words=vocab_size, oov_token="UNK")
  context_tokenizer.fit_on_texts(traincontext)

  # Train
  # set up config and create model
  exit()
  mdl = AttentionGRUModel(config)
  model = mdl.create_model()
  exit() # works until here


  model = Sequential()

  # Using GLOVE word embeddings
  # Source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
  embeddings_index = {}
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

  model.add(Embedding(vocab_size, vector_size, weights=[embedding_matrix], input_length=text_maxlen, trainable=False))
  model.add(LSTM(vector_size, return_sequences=True))
  model.add(Flatten())
  model.add(Dropout(0.3))
  model.add(Dense(num_classes, activation='softmax'))

  model.summary()

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  K.set_value(model.optimizer.learning_rate, 0.001)

  history = model.fit(Xtrain, Ytrain,
                      batch_size=batch_size,
                      epochs=38,
                      verbose=1,
                      validation_data=(Xval, Yval))

  # Save tokenizer
  tokenizer_json = tokenizer.to_json()
  with io.open('qasys/toks/qa_tok.json', 'w', encoding='utf-8') as f:
      f.write(tokenizer_json)

  # Save model
  model.save('qasys/data/qa_g_lstm.h5')

  if MODE == 1:
    exit()

  Ypred = model.predict(Xtest)
  Ypred = np.argmax(Ypred, axis=1)
  Ytest = np.argmax(Ytest, axis=1)

  with open('eval_report.txt', 'w') as f:
    sys.stdout = f
    print(metrics.classification_report(Ytest, Ypred, target_names=type_descs))

  cm = metrics.confusion_matrix(Ytest, Ypred).transpose()
  np.set_printoptions(linewidth=500)
  np.set_printoptions(threshold=np.inf)
  with open('eval_confusionmatrix.txt', 'w') as f:
    sys.stdout = f
    print(cm)

if __name__ == '__main__':
  train_main()