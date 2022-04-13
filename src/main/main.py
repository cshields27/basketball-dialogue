'''
Main driver for Basketball Dialogue System

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''
import numpy as np
import tensorflow as tf
import sys

from keras.preprocessing.text import tokenizer_from_json

C_TOK_PATH = '../../data/context_tok.json'
A_TOK_PATH = '../../data/answer_tok.json'
Q_TOK_PATH = '../../data/question_tok.json'

def read_question():
  try:
    question = str(sys.argv[2])
  except:
    print("No question found: use -q flag before question")
    exit()
  return question

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
  model = tf.keras.models.load_model('simple_qa_model.h5')
  return model

def prediction(contexts_tok, answers_tok, tokenized_question, model):
  # For now just the first context
  tokenized_context = contexts_tok.texts_to_sequences([0])
  tokenized_answer = answers_tok.texts_to_sequences(['<s>'])

  tokenized_context = np.array(tokenized_context, dtype=np.float)
  tokenized_answer = np.array(tokenized_answer, dtype=np.float)
  tokenized_question = np.array(tokenized_question, dtype=np.float)

  # Loop through and predict next word in answer sequence
  for i in range(1, len(tokenized_answer[0])):
    prediction = model.predict([tokenized_context, tokenized_answer, tokenized_question])
    next_word = np.argmax(prediction)
    tokenized_answer[0][i] = next_word
    detokenize_display(answers_tok, tokenized_answer)

def detokenize_display(answers_tok, tokenized_answer):
  answer_seq = answers_tok.sequences_to_texts(tokenized_answer)
  answer_str = answer_seq[0].replace('<NULL>', '')
  print(answer_str)

def main():
  print('Welcome to the IDS Spring 2022 NBA Dialogue System version 1.0')
  print('Getting things ready...')
  question = read_question()
  contexts_tok, answers_tok, questions_tok = load_tokenizers()
  tokenized_question = tok_question(questions_tok, question)
  model = load_model()
  prediction(contexts_tok, answers_tok, tokenized_question, model)
    
if __name__ == '__main__':
  main()