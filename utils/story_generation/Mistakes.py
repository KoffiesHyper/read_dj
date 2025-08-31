import random
import json

from decouple import config

SG_PATH = config("SG_PATH")

def get_mistakes(mistakes):
    
    errors = [[word, mistakes[word]] for word in mistakes.keys()]

    with open(f"{SG_PATH}/phoneme_vocab_display.json", "r") as f:
        vocab = json.loads(f.read()).keys()
        vocab = [word for word in vocab]

    for i in range(5 - len(errors)):
        random_i = random.randint(0, len(vocab))

        while vocab[random_i] in mistakes:
            random_i = random.randint(0, len(vocab))
        
        errors.append([vocab[random_i], random.randint(1, 10)])        
        
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