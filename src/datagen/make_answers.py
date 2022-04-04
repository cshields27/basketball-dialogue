'''
Script to generate answers to several questions for each NBA player

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate

Also may need to install nba_api:
pip install nba_api
'''

import csv
import requests
import time
from bs4 import BeautifulSoup
import pandas as pd

PLAYER_NAMES_PATH = '../../data/players.csv'

TEAM_IDS_TO_NAMES = {
'ATL'	: 'Atlanta Hawks',
'BKN'	: 'Brooklyn Nets',
'BOS'	: 'Boston Celtics',
'CHA'	: 'Charlotte Hornets',
'CHI'	: 'Chicago Bulls',
'CLE'	: 'Cleveland Cavaliers',
'DAL'	: 'Dallas Mavericks',
'DEN'	: 'Denver Nuggets',
'DET'	: 'Detroit Pistons',
'GSW'	: 'Golden State Warriors',
'HOU' : 'Houston Rockets',
'IND'	: 'Indiana Pacers',
'LAC'	: 'Los Angeles Clippers',
'LAL'	: 'Los Angeles Lakers',
'MEM'	: 'Memphis Grizzlies',
'MIA'	: 'Miami Heat',
'MIL'	: 'Milwaukee Bucks',
'MIN'	: 'Minnesota Timberwolves',
'NOP'	: 'New Orleans Pelicans',
'NYK'	: 'New York Knicks',
'OKC'	: 'Oklahoma City Thunder',
'ORL'	: 'Orlando Magic',
'PHI'	: 'Philadelphia 76ers',
'PHX'	: 'Phoenix Suns',
'POR' : 'Portland Trail Blazers',
'SAC'	: 'Sacramento Kings',
'SAS'	: 'San Antonio Spurs',
'TOR' : 'Toronto Raptors',
'UTA'	: 'Utah Jazz',
'WAS'	: 'Washington Wizards'
} 

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

def generate_answers():
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
    if team and city:
      print(f'{player["full_name"]} {city} {team}')    

def main():
  generate_answers()

if __name__ == '__main__':
  main()