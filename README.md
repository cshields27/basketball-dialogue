#basketball-dialogue
Connor Shields and Kenan Lumantas
Project for CSE 40982

See attached .docx for general overview.
data/ contains generated question, answer, context files; pickle files for training; the model itself; (these were left in submission but can all be safely deleted as they are large)
src/ contains the main driver (which is the only thing users need to run) as well as a datagen folder for making contexts, questions/answers, and training the model

the train model script has DEBUG and TRAIN modes which should be toggled on and off. in DEBUG mode, very small test files are used. if TRAIN == False, 
the script simply preprocesses and saves pickle files of the data. if TRAIN == True, pickle files are loaded and trained on. train model consumes large amounts of memory and/or disk
depending on settings

main.py loads the model and requests user question and context inputs. it generates answers using this input.

We hope you enjoy playing around with the system!
