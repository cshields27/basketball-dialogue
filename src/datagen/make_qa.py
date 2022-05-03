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
CONTEXTS_WRITE = '../../data/contexts.test'
QUESTIONS_PATH = '../../data/questions.test'
ANSWERS_PATH = '../../data/answers.test'

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

QPSCH = '../../data/questions_school.test'
APSCH = '../../data/answers_school.test'

QPYR = '../../data/questions_year.test'
APYR = '../../data/answers_year.test'

QPPOS = '../../data/questions_position.test'
APPOS = '../../data/answers_position.test'

def answer_stats_question(stat_str, stat, question_temps, name, status, season, f_q, f_a, fc, context):
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
    fc.write(context)

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

  fc = open(CONTEXTS_WRITE, 'w')
  fq = open(QUESTIONS_PATH, 'w')
  fa = open(ANSWERS_PATH, 'w')
  
  lines = f_c.readlines()

  for line in lines:
    context = line
    line = line[9:-7]
    player_info = eval(line)
    name = player_info['info']['DISPLAY_FIRST_LAST']
    city = player_info['info']['TEAM_CITY']
    team = player_info['info']['TEAM_NAME']
    status = player_info['info']['ROSTERSTATUS']
    school = player_info['info']['SCHOOL']
    years = player_info['info']['SEASON_EXP']
    position = player_info['info']['POSITION']
    ppg = player_info['averages']['PTS']
    rpg = player_info['averages']['REB']
    apg = player_info['averages']['AST']
    career_data = player_info['career_data']
    #allstar_appearances = int(player_info['averages']['ALL_STAR_APPEARANCES'])
    allstar_appearances = 0
    season = player_info['averages']['TimeFrame']
    

    ''' Answer player team question '''
    for question_temp in types_to_qs['player_team']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        answer = f'{name} played for the {city} {team}.'
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
      else:
        answer = f'{name} plays for the {city} {team}.'
      
      fq.write('STARTTAG {} ENDTAG\n'.format(question))
      fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      fc.write(context)

    ''' Answer each stats question '''
    answer_stats_question('points', ppg, types_to_qs['player_points'], name, status, season, fq, fa, fc, context)
    answer_stats_question('assists', apg, types_to_qs['player_assists'], name, status, season, fq, fa, fc, context)
    answer_stats_question('rebounds', rpg, types_to_qs['player_rebounds'], name, status, season, fq, fa, fc, context)

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

      fq.write('STARTTAG {} ENDTAG\n'.format(question))
      fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      fc.write(context)

    ''' Answer All-Star game question '''
    for question_temp in types_to_qs['player_allstar']:
      question = question_temp.replace('_', name)

      has = '' if status == 'Inactive' else 'has '
      if allstar_appearances:
        s = 's' if allstar_appearances > 1 else ''
        answer = f'{name} {has}made {allstar_appearances} All-Star team{s}'
      else:
        answer = f'{name} {has}never made an All-Star team'

      #fq.write('STARTTAG {} ENDTAG\n'.format(question))
      #fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      #fc.write(context)

    ''' Answer school question '''
    for question_temp in types_to_qs['player_school']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
        question = question.replace('are', 'were')
      answer = f'{name} played at {school} for college.'

      # fqsch.write(f'STARTTAG {question} ENDTAG\n')
      # fasch.write(f'STARTTAG {answer} ENDTAG\n')
      fq.write('STARTTAG {} ENDTAG\n'.format(question))
      fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      fc.write(context)

    ''' Answer years question '''
    for question_temp in types_to_qs['player_years']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
        question = question.replace('are', 'were')
        answer = f'{name} played for {years} years.'
      else:
        answer = f'{name} has been playing for {years} years.'

      fq.write('STARTTAG {} ENDTAG\n'.format(question))
      fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      fc.write(context)

    ''' Answer position question '''
    for question_temp in types_to_qs['player_position']:
      question = question_temp.replace('_', name)

      if status == 'Inactive':
        question = question.replace('is', 'was')
        question = question.replace('does', 'did')
        question = question.replace('are', 'were')
        answer = f'{name} played {position}.'
      else:
        answer = f'{name} plays {position}.'

      fq.write('STARTTAG {} ENDTAG\n'.format(question))
      fa.write('STARTTAG {} ENDTAG\n'.format(answer))
      fc.write(context)

    ''' Answer best season question '''
    for question_temp in types_to_qs['player_best_season']:

      question = question_temp.replace('_', name)
      best_season = 0
      best_season_total = 0
      for season, stats in career_data.items():
        # let's just use a really simple metric to quantify 'best' season
        for key in stats:
          stats[key] = stats[key] if stats[key] else 0

        total = stats['PTS'] + stats['REB'] + stats['AST'] + stats['BLK'] + stats['FG_PCT']*100 - stats['TOV']
        if total >= best_season_total:
          best_season = season
          best_season_total = total
        
      if best_season != 0:
        for key in career_data[best_season]:
          career_data[best_season][key] = career_data[best_season][key] if career_data[best_season][key] else 0

        best_ppg = str(round(career_data[best_season]['PTS']/career_data[best_season]['GP'], 2))
        best_rpg = str(round(career_data[best_season]['REB']/career_data[best_season]['GP'], 2))
        best_apg = str(round(career_data[best_season]['AST']/career_data[best_season]['GP'], 2))
        
        answer = f"{name} had his best season in {best_season}. He averaged {best_ppg} points, {best_rpg} rebounds, and {best_apg} assists."  
        fq.write('STARTTAG {} ENDTAG\n'.format(question))
        fa.write('STARTTAG {} ENDTAG\n'.format(answer))
        fc.write(context)

  fq.close()
  fa.close()
  fc.close()


def main():
  types_to_qs = generate_questions()
  generate_qa(types_to_qs)

if __name__ == '__main__':
  main()
