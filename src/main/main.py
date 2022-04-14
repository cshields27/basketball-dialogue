'''
Main driver for Basketball Dialogue System

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''
import numpy as np
import tensorflow as tf
import re
import sys

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

CONTEXTS_PATH = '../../data/contexts.test'

C_TOK_PATH = '../../data/context_tok.json'
A_TOK_PATH = '../../data/answer_tok.json'
Q_TOK_PATH = '../../data/question_tok.json'
MODEL_PATH = '../../data/model.h5'

def read_question():
  try:
    question = str(sys.argv[2])
  except:
    print("No question found: use -q flag before question")
    exit()
  return question

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

def get_context(path):
  lines = []
  with open(path) as f:
    for line in f.readlines():
      return preprocess(line)
  return lines

def load_tokenizers():
  with open(C_TOK_PATH) as f:
    contexts_data = f.read()
    contexts_tok = tokenizer_from_json(contexts_data)

  with open(A_TOK_PATH) as f:
    answers_data = f.read()
    answers_tok = tokenizer_from_json(answers_data)

  with open(Q_TOK_PATH) as f:
    questions_data = f.read()
    questions_tok = tokenizer_from_json(questions_data)

  return contexts_tok, answers_tok, questions_tok

def tok_question(questions_tok, question):
  tokenized_question = questions_tok.texts_to_sequences([question])
  return tokenized_question

def load_model():
  model = tf.keras.models.load_model(MODEL_PATH)
  return model

def get_prediction(context, contexts_tok, answers_tok, tokenized_question, model):
  # note - we might need to add spaces before/after the <s> so that they get picked up when tokenizing; right now there are no spaces
  context_tokenization = contexts_tok.texts_to_sequences([context])
  context_tokenization = pad_sequences(context_tokenization, padding="post", truncating="post", maxlen=1000)
  prediction = answers_tok.texts_to_sequences(['<s>'])
  prediction = pad_sequences(prediction, padding="post", truncating="post", maxlen=10)
  question_tokenization = pad_sequences(tokenized_question, padding="post", truncating="post", maxlen=10)

  word_num = 1
  while True:
    out = model.predict((np.asarray(question_tokenization), np.asarray(prediction), np.asarray(context_tokenization)))
    predict_index = np.argmax(out[0]) # find the max value in the output prediction
    prediction[0][word_num] = predict_index # add the max index to our prediction
    next_word = answers_tok.sequences_to_texts([[predict_index]])
    if next_word == ['</s>']: # exit condition
      return answers_tok.sequences_to_texts(prediction)
    print(answers_tok.sequences_to_texts(prediction))
    word_num += 1

def detokenize_display(answers_tok, tokenized_answer):
  answer_seq = answers_tok.sequences_to_texts(tokenized_answer)
  answer_str = answer_seq[0].replace('<NULL>', '')
  print(answer_str)

def main():
  print('Welcome to the IDS Spring 2022 NBA Dialogue System version 1.0')
  print('Getting things ready...')
  question = read_question()
  contexts_tok, answers_tok, questions_tok = load_tokenizers()
  tokenized_question = tok_question(questions_tok, preprocess(question, True))
  model = load_model()
  print(get_prediction(get_context(CONTEXTS_PATH), contexts_tok, answers_tok, tokenized_question, model))
    
if __name__ == '__main__':
  main()