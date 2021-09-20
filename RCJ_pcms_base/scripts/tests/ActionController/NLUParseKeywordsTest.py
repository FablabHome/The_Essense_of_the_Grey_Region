#!/usr/bin/env python3
import json

from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

with open('./dataset.json') as f:
    sample_dataset = json.load(f)

nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
nlu_engine = nlu_engine.fit(sample_dataset)
try:
    while True:
        text = input('Your command: \n|__ ')
        parsing = nlu_engine.parse(text)
        print(parsing)
except KeyboardInterrupt:
    print('Program ended')
