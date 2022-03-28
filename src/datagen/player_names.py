'''
Script to generate CSV list of all NBA players and the URL to their page

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''

import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_all_names():
  alphabet = 'abcdefghijklmnopqrstuvwxyz'

  # loop over every starting letter
  for letter in alphabet:
    url = f'https://www.basketball-reference.com/players/{letter}/'
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find(class_="stats_table")

    # access table body
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')

    for trb in tr_body:
      # process row
      th = trb.find('th')
      name = th.get_text().replace('*', '')
      name_url = f"https://www.basketball-reference.com{th.find('a').get('href')}"
      yield (name, name_url)

def create_player_csv():
  with open('../../data/players.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'url'])
    for name, url in get_all_names():
      writer.writerow([name, url])

def main():
  create_player_csv()

if __name__ == '__main__':
  main()