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

QPTEAM = '../../data/questions_team.test'
APTEAM = '../../data/answers_team.test'

QPAVG = '../../data/questions_averages.test'
APAVG = '../../data/answers_averages.test'

QPPPG = '../../data/questions_points.test'
APPPG = '../../data/answers_points.test'

QPAPG = '../../data/questions_assists.test'
APAPG = '../../data/answers_assists.test'

QPRPG = '../../data/questions_rebounds.test'
APRPG = '../../data/answers_rebounds.test'

QPASG = '../../data/questions_allstar.test'
APASG = '../../data/answers_allstar.test'

def answer_stats_question(stat_str, stat, question_temps, name, status, season, f_q, f_a):
  ''' Answers a question for stats such as ppg '''

  for question_temp in question_temps:
    question = question_temp.replace('_', name)

    if status == 'Inactive':
      question = question.replace('is', 'was')
      question = question.replace('does', 'did')
      question = question.replace('are', 'were')
      answer = f'{name} averaged {stat} {stat_str} per game in his career.'
    else:
      answer = f'{name} averaged {stat} {stat_str} per game in the {season} season.'

    f_q.write('STARTTAG {} ENDTAG\n'.format(question))
    f_a.write('STARTTAG {} ENDTAG\n'.format(answer))

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

  fqteam = open(QPTEAM, 'w')
  fateam = open(APTEAM, 'w')

  fqavg = open(QPAVG, 'w')
  faavg = open(APAVG, 'w')

  fqppg = open(QPPPG, 'w')
  fappg = open(APPPG, 'w')

  fqrpg = open(QPRPG, 'w')
  farpg = open(APRPG, 'w')

  fqapg = open(QPAPG, 'w')
  faapg = open(APAPG, 'w')

  fqasg = open(QPASG, 'w')
  faasg = open(APASG, 'w')

  
  lines = f_c.readlines()

  for line in lines:
    line = line[9:-7]
    player_info = eval(line)
    name = player_info['info']['DISPLAY_FIRST_LAST']
    city = player_info['info']['TEAM_CITY']
    team = player_info['info']['TEAM_NAME']
    status = player_info['info']['ROSTERSTATUS']
    ppg = player_info['averages']['PTS']
    rpg = player_info['averages']['REB']
    apg = player_info['averages']['AST']
    #allstar_appearances = int(player_info['averages']['ALL_STAR_APPEARANCES'])
    allstar_appearances = 0
    season = player_info['averages']['TimeFrame']

    ''' Answer player team question '''
    for question_temp in types_to_qs['player_team']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        answer = f'He played for the {city} {team}.'
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
      else:
        answer = f'He plays for the {city} {team}.'
      
      fqteam.write('STARTTAG {} ENDTAG\n'.format(question))
      fateam.write('STARTTAG {} ENDTAG\n'.format(answer))

    ''' Answer each stats question '''
    answer_stats_question('points', ppg, types_to_qs['player_points'], name, status, season, fqppg, fappg)
    answer_stats_question('assists', apg, types_to_qs['player_assists'], name, status, season, fqapg, faapg)
    answer_stats_question('rebounds', rpg, types_to_qs['player_rebounds'], name, status, season, fqrpg, farpg)

    ''' Answer averages question '''
    for question_temp in types_to_qs['player_averages']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
        question = question.replace('are', 'were')
        answer = f'{name} averaged {ppg} points per game, {apg} assists per game, and {rpg} rebounds per game in his career.'
      else:
        answer = f'{name} averaged {ppg} points per game, {apg} assists per game, and {rpg} rebounds per game in the {season} season.'

      fqavg.write('STARTTAG {} ENDTAG\n'.format(question))
      faavg.write('STARTTAG {} ENDTAG\n'.format(answer))

    ''' Answer All-Star game question '''
    for question_temp in types_to_qs['player_allstar']:
      question = question_temp.replace('_', name)

      has = '' if status == 'Inactive' else 'has '
      if allstar_appearances:
        s = 's' if allstar_appearances > 1 else ''
        answer = f'{name} {has}made {allstar_appearances} All-Star team{s}'
      else:
        answer = f'{name} {has}never made an All-Star team'

      fqasg.write(f'STARTTAG {question} ENDTAG\n')
      faasg.write(f'STARTTAG {answer} ENDTAG\n')

  fqteam.close()
  fateam.close()

  fqavg.close()
  faavg.close()

  fqppg.close()
  fappg.close()
  fqrpg.close()
  farpg.close()
  fqapg.close()
  faapg.close()

  fqasg.close()
  faasg.close()


def main():
  types_to_qs = generate_questions()
  generate_qa(types_to_qs)

if __name__ == '__main__':
  main()
