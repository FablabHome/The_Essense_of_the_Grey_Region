#!/usr/bin/env python3

from snips_nlu import SnipsNLUEngine

nlu_engine = SnipsNLUEngine.from_path('./test_engine')
try:
    while True:
        text = input('Your command: \n|__ ')
        parsing = nlu_engine.parse(text)
        print(parsing)
except KeyboardInterrupt:
    print('Program ended')
