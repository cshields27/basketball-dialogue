'''
Script to generate answers to several questions for each NBA player

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''

import csv
import requests
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

def get_stat(soup, tag):
  table = soup.find(class_="stats_table")
  tbody = table.find('tbody')
  tr_body = tbody.find_all('tr')[-1]
  
  for trb in tr_body:
    # process row
    if tag == 'team_id' and trb.get_text() in TEAM_IDS_TO_NAMES:
      print(TEAM_IDS_TO_NAMES[trb.get_text()])

def generate_answers():
  # Process each player
  with open(PLAYER_NAMES_PATH, 'r') as f:
    for i, line in enumerate(f.readlines()):
      if i == 0: continue
      name, link = line.strip().split(',')
      print(name, link)
      response = requests.get(link).text
      soup = BeautifulSoup(response, 'html.parser')
      team_answer = get_stat(soup, 'team_id')
      if i > 5:
        exit()

def main():
  generate_answers()

if __name__ == '__main__':
  main()