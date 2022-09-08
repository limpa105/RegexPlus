import random, string, pickle
from typing import *
import regex_probs

MAX_LENGTH = 25

special_regex_components = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
chars = string.printable[:95]

def random_regex() -> List[str]:
    length = random.choice(range(MAX_LENGTH))
    return [random.choice(special_regex_components if random.random() < 0.4 else chars) for i in range(length)]


NUM_TRAIN_REGEXES = 50000 # 50k
NUM_SAMPLES = 5
NUM_TEST_REGEXES = 1000

def main():
    train_data = []
    for i in range(NUM_TRAIN_REGEXES):
        regex = random_regex()
        r = regex_probs.Regex(regex)
        examples = [r.sample() for i in range(NUM_SAMPLES)]
        train_data.append((regex, examples))
        if (i % 3000) == 0: print(i)
    test_regexes = [random_regex() for i in range(NUM_TEST_REGEXES)]
    pickle.dump((train_data, test_regexes), open('hmmmmm.pickle', 'wb'))

if __name__ == '__main__':
    main()

