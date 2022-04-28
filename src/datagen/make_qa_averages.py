'''
Script to generate questions and answers to from contexts.test for each NBA player

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate

Also may need to install nba_api:
pip install nba_api
'''

from socket import timeout
import time
import pandas as pd
from collections import defaultdict

QUESTION_TEMPLATES_PATH = '../../data/questiontemplates.txt'
CONTEXTS_PATH = '../../data/contexts_original.test'
QUESTIONS_PATH = '../../data/questions_averages.test'
ANSWERS_PATH = '../../data/answers_averages.test'

def generate_questions():
  ''' Read questions templates into a dict mapping type to templates '''
  types_to_qs = defaultdict(list)
  with open(QUESTION_TEMPLATES_PATH, 'r') as f:
    for line in f.readlines():
      q_type, q = line.strip().split(',', 1)
      types_to_qs[q_type].append(q)
  return types_to_qs

def generate_qa(types_to_qs):
  # Open file
  f_c = open(CONTEXTS_PATH, 'r')
  f_q = open(QUESTIONS_PATH, 'w')
  f_a = open(ANSWERS_PATH, 'w')
  
  lines = f_c.readlines()

  for line in lines[::5]:
    player_info = eval(line)
    name = player_info['info']['DISPLAY_FIRST_LAST']
    status = player_info['info']['ROSTERSTATUS']
    ppg = player_info['averages']['PTS']
    apg = player_info['averages']['AST']
    rpg = player_info['averages']['REB']
    season = player_info['averages']['TimeFrame']

    for question_temp in types_to_qs['player_averages']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
        question = question.replace('are', 'were')
        answer = name + ' averaged ' + ppg + ' points per game, ' + apg + ' assists per game, and ' + rpg + ' rebounds per game in their career.'
      else:
        answer = name + ' averaged ' + ppg + ' points per game, ' + apg + ' assists per game, and ' + rpg + ' rebounds per game in the ' + season + ' season.'

      f_q.write('STARTTAG {} ENDTAG\n'.format(question))
      f_a.write('STARTTAG {} ENDTAG\n'.format(answer))

def main():
  types_to_qs = generate_questions()
  generate_qa(types_to_qs)

if __name__ == '__main__':
  main()
