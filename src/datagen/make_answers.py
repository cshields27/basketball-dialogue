'''
Script to generate answers to several questions for each NBA player

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate

Also may need to install nba_api:
pip install nba_api
'''

import csv
import random
import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict

PLAYER_NAMES_PATH = '../../data/players.csv'
QUESTIONS_PATH = '../../data/questiontemplates.txt'

import pickle

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

def generate_questions():
  ''' Read questions templates into a dict mapping type to templates'''
  types_to_qs = defaultdict(list)
  with open(QUESTIONS_PATH, 'r') as f:
    for line in f.readlines():
      q_type, q = line.strip().split(',', 1)
      types_to_qs[q_type].append(q)
  return types_to_qs

def generate_answers(types_to_qs):
  # Process each player
  for player in players.get_players():
    # Get info through API
    time.sleep(.600) # sleep so the api/NBA stats doesn't kick us out
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
    player_common = player_info.common_player_info.get_dict()

    # Get team and city
    team_ind = player_common['headers'].index('TEAM_NAME')
    city_ind = player_common['headers'].index('TEAM_CITY')
    team = player_common['data'][0][team_ind]
    city = player_common['data'][0][city_ind]

    # Generate question for team by randomly picking from list
    q_template = random.choice(types_to_qs['player_team'])
    question = q_template.replace('_', player['full_name'])

    if team and city:
      print(f'{question} {city} {team}')    

def main():
  types_to_qs = generate_questions()
  generate_answers(types_to_qs)

if __name__ == '__main__':
  main()