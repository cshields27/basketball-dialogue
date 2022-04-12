'''
Main driver for Basketball Dialogue System

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''

def read_question():
  return input('- ')

def main():
  print('Welcome to the IDS Spring 2022 NBA Dialogue System version 1.0')
  print('Getting things ready...')
  while (question := read_question()):
    pass
    
if __name__ == '__main__':
  main()