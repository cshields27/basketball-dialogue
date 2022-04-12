'''
Generate file for context where answers will be derived from

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate

Also may need to install nba_api:
pip install nba_api
'''

import csv
import random
from socket import timeout
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

def generate_contexts():
  # Open file
  f = open('../../data/contexts.test', 'w')

  # Go through each player
  for player in players.get_players():
    time.sleep(.5) # sleep so the api/NBA stats doesn't kick us out
    # Access API endpoints
    common_player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'], timeout=100)
    career_stats = playercareerstats.PlayerCareerStats(player_id=player['id'])

    # Get common player info as well as regular season stats
    player_info = common_player_info.common_player_info.get_dict()
    reg_season_stats = career_stats.season_totals_regular_season.get_dict()
    # print(player_info)
    # print(reg_season_stats)

    # Write contexts to contexts.test 
    # Get player info and stats dicts
    player = dict()
    player_info = dict(zip(player_info['headers'], player_info['data'][0]))
    career_data = dict()

    # Get player stats for every year
    for year_data in reg_season_stats['data']:
      year_data = dict(zip(reg_season_stats['headers'], year_data))
      career_data[year_data['SEASON_ID']] = year_data

    # Write dict to contexts file
    player['info'] = player_info
    player['career_data'] = career_data

    f.write('{}\n'.format(str(player)))

    name = player_info['DISPLAY_FIRST_LAST']

    print(name, 'added to contexts.test')

  f.close()

def main():
  generate_contexts()

if __name__ == '__main__':
  main()