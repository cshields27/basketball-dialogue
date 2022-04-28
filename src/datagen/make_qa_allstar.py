'''
Script to generate questions and answers to from contexts.test for each NBA player
Answers ALL STAR question

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
QUESTIONS_PATH = '../../data/questions_allstar.test'
ANSWERS_PATH = '../../data/answers_allstar.test'

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
    allstar_appearances = int(player_info['averages']['ALL_STAR_APPEARANCES'])

    for question_temp in types_to_qs['player_allstar']:
      question = question_temp.replace('_', name)

      has = '' if status == 'Inactive' else 'has '
      if allstar_appearances:
        s = 's' if allstar_appearances > 1 else ''
        answer = f'{name} {has}made {allstar_appearances} All-Star team{s}'
      else:
        answer = f'{name} {has}never made an All-Star team'

      f_q.write(f'STARTTAG {question} ENDTAG\n')
      f_a.write(f'STARTTAG {answer} ENDTAG\n')

def main():
  types_to_qs = generate_questions()
  generate_qa(types_to_qs)

if __name__ == '__main__':
  main()
