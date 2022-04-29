'''
Pull from several files to create our training data files

Run to activate venv:
source /escnfs/home/cmc/public/venv/bin/activate
'''

TRAIN_QUESTION_PATH = '../../data/trainquestion.txt'
TRAIN_CONTEXT_PATH = '../../data/traincontext.txt'
TRAIN_ANSWER_PATH = '../../data/trainanswer.txt'

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

CONTEXTS_PATH = '../../data/contexts_original.test'

questions_files = [QPTEAM, QPAVG, QPPPG, QPAPG, QPRPG]
answers_files = [APTEAM, APAVG, APPPG, APAPG, APRPG]

def make_dataset():
  tq_p = open(TRAIN_QUESTION_PATH, 'w')
  tc_p = open(TRAIN_CONTEXT_PATH, 'w')
  ta_p = open(TRAIN_ANSWER_PATH, 'w')

  contexts = []
  with open(CONTEXTS_PATH, 'r') as f:
    contexts = f.readlines()


  for i in range(len(questions_files)):
    q_file = open(questions_files[i], 'r')
    a_file = open(answers_files[i], 'r')

    # read from each file, append to overall training set
    questions = q_file.readlines()
    answers = a_file.readlines()
    for j in range(0, len(questions), 51):
      tq_p.write(questions[j])
      tc_p.write(contexts[j//5])
      ta_p.write(answers[j])

    q_file.close()
    a_file.close()


if __name__ == '__main__':
  make_dataset()