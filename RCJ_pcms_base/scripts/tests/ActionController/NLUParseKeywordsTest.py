#!/usr/bin/env python3
from core.utils.KeywordParsers import HeySnipsNLUParser

introduce_dialog = '''
Ah, Forgive me for not introducing myself, masters.
I'm snippy, your virtual assistant in this restaurant,
I'm still under development, so you could only see me talking
right now.
'''

nlu_engine = HeySnipsNLUParser(dataset_path='datasets/orderfood/dataset.json')

print('Greetings! Welcome to snips restaurant, let us warm your day!')
try:
    while True:
        text = input('Your command: \n|__ ')
        user_intent, intent_probability, slots = nlu_engine.parse(text)

        print(f'User intent: {user_intent}, probability: {intent_probability}')
        print('Parsed slots:')

        for idx, slot in enumerate(slots):
            entity = slot['entity']
            raw_value = slot['rawValue']
            print(f'Slot {idx + 1}:')
            print(f'\tentity: {entity}, rawValue: {raw_value}')

        print()
        if user_intent == 'Introduce' and intent_probability >= 0.43:
            print(introduce_dialog)

except KeyboardInterrupt:
    print('Program ended')
