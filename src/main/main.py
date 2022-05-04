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

CONTEXTS_PATH = '../../data/contexts_original.test'

C_TOK_PATH = '../../data/context_tok.json'
A_TOK_PATH = '../../data/answer_tok.json'
Q_TOK_PATH = '../../data/question_tok.json'
MODEL_PATH = '../../data/model_running.h5'

def read_question():
  try:
    user_in = input('- ')
  except:
    exit()

  try:
    question, cline = user_in.rsplit(',', 1)
    cline = int(cline)
    if cline <= 0:
      cline = None
    return question, cline
  except:
    return user_in, None

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

def get_context(path, cline):
  lines = []
  with open(path) as f:
    return f.readlines()[cline - 1]

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
    #questions_tok.filters = ''

  return contexts_tok, answers_tok, questions_tok

def tok_question(questions_tok, question):
  question = f'STARTTAG {question} ENDTAG'
  question = preprocess(question, True)
  tokenized_question = questions_tok.texts_to_sequences([question])
  tokenized_question = pad_sequences(tokenized_question, padding="post", truncating="post", maxlen=20)
  return tokenized_question

def load_model():
  model = tf.keras.models.load_model(MODEL_PATH)
  return model

def get_prediction(context, contexts_tok, answers_tok, tokenized_question, model):
  context_tokenization = contexts_tok.texts_to_sequences([preprocess(context)])
  context_tokenization = pad_sequences(context_tokenization, padding="post", truncating="post", maxlen=1000)
  prediction = answers_tok.texts_to_sequences(['STARTTAG'])
  prediction = pad_sequences(prediction, padding="post", truncating="post", maxlen=20)
  question_tokenization = pad_sequences(tokenized_question, padding="post", truncating="post", maxlen=20)
  
  word_num = 1
  while True:
    out = model.predict((np.asarray(question_tokenization), np.asarray(prediction), np.asarray(context_tokenization)))
    predict_index = np.argmax(out[0]) # find the max value in the output prediction
    prediction[0][word_num] = predict_index # add the max index to our prediction
    next_word = answers_tok.sequences_to_texts([[predict_index]])

    if next_word == ['ENDTAG'] or word_num == 9: # exit condition
      #return answers_tok.sequences_to_texts(prediction)
      return prediction
    word_num += 1

def detokenize_display(answers_tok, tokenized_answer):
  answer_seq = answers_tok.sequences_to_texts(tokenized_answer)
  answer_str = answer_seq[0].replace('STARTTAG', '')
  answer_str = answer_str.replace('ENDTAG', '')
  answer_str = answer_str.replace('UNK', '')
  answer_str = answer_str.strip()
  print(answer_str)

def main():
  print('Welcome to the IDS Spring 2022 NBA Dialogue System version 1.0')
  print('Getting things ready...')
  contexts_tok, answers_tok, questions_tok = load_tokenizers()
  model = load_model()
  print('Ok, you can begin asking questions now!')
  print('Input a question, followed by a comma and the line of context desired (from data/contexts_original.test)')
  print('If no context is provided, the most recent one will be used')

  question = None
  cline = None
  while cline is None:
    question, cline = read_question()
  cline = int(cline)

  while question:
    tokenized_question = tok_question(questions_tok, question)
    prediction = get_prediction(get_context(CONTEXTS_PATH, cline), contexts_tok, answers_tok, tokenized_question, model)
    detokenize_display(answers_tok, prediction)

    # Get next question
    question, nextcline = read_question()
    if nextcline is not None:
      cline = int(nextcline)
    
if __name__ == '__main__':
  main()
