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
from nba_api.stats.endpoints import playercareerstats

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
    # Access API endpoints
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
    career_stats = playercareerstats.PlayerCareerStats(player_id=player['id'])

    # Get commons player info as well as regular season stats
    player_common = player_info.common_player_info.get_dict()
    regular_season_stats = career_stats.season_totals_regular_season.get_dict() # TODO need to use this still

    # Get team and city
    keys_to_stats = dict(zip(player_common['headers'], player_common['data'][0]))

    team = keys_to_stats['TEAM_NAME']
    city = keys_to_stats['TEAM_CITY']
    first_year = keys_to_stats['FROM_YEAR']
    last_year = keys_to_stats['TO_YEAR']
    status = keys_to_stats['ROSTERSTATUS']

    # Generate question for team by randomly picking from list
    q_template = random.choice(types_to_qs['player_team'])
    question = q_template.replace('_', player['full_name'])

    # If player isn't active, switch up the question
    if status == 'Inactive':
      question = question.replace('is', 'was')
      question = question.replace('does', 'did')

    if team and city:
      print(f'{question} {city} {team}')    

def main():
  types_to_qs = generate_questions()
  generate_answers(types_to_qs)

if __name__ == '__main__':
  main()