import random

from decouple import config

SG_PATH = config("SG_PATH")

def get_mistakes(mistakes):
        
    errors = [[word, str(random.randint(1, 10))] for word in mistakes.keys()]    
        
    maxi = 0
    max_word = ""
    delete = 0
    output = ""

    for j in range(5):
        for i in range(len(errors)):
            number = int(errors[i][1])
            if number > maxi:
                maxi = number
                max_word = errors[i][0]
                delete = i
        output = output +  max_word + " " 
        maxi = 0
        del errors[delete]
        delete = 0
        max_word = ""
        
    with open(f"{SG_PATH}/phonemes.txt", 'w', encoding='utf-8') as f:
        f.write(output.strip())

# get_mistakes("phoneme_vocab_display.json")