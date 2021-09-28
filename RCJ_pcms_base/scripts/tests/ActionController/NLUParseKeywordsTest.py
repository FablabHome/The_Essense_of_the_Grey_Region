#!/usr/bin/env python3
from core.utils.KeywordParsers import HeySnipsNLUParser

introduce_dialog = '''
Ah, Forgive me for not introducing myself, masters.
I'm snippy, your virtual assistant in this restaurant,
I'm still under development, so you could only see me talking
right now.
'''

menu = '''
Menu                          Price
-------------------------------------
French Fries                    $7
meat salad                     $20
spaghetti                      $23
hot chocolate                  $14
cappucino                      $19
tea                             $0
water                           $0
Hamburger                      $19
Ketchup                         $0
Tacos                          $15
Marshmellos                    $10
Steak                          $27
'''

datasets = {
    'restaurant': 'datasets/orderfood/dataset.json',
    # 'home': 'datasets/getWeather/dataset.json'
}
parser = HeySnipsNLUParser(dataset_configs=datasets)

print('Greetings! Welcome to snips restaurant, let us warm your day!')
try:
    while True:
        user_intent = ''
        intent_probability = 0.0
        text = input('Your command: \n|__ ')
        result = list(parser.parse_full_data(text))
        print('')
        print('\n--------------------------\n| Engine Parsing Results |')
        for engine_id, user_intent, intent_probability, slots in result:
            print('-------------------------------------------------------------')
            print(f'Parse result of engine "{engine_id}"')
            print(f'User intent: {user_intent}, probability: {intent_probability}')
            print('Parsed slots:')

            for idx, slot in enumerate(slots):
                entity = slot['entity']
                raw_value = slot['rawValue']
                print(f'Slot {idx + 1}:')
                print(f'\tentity: {entity}, rawValue: {raw_value}')

            print()

        print('-------------------------------------------------------------')
        print('\n***************************\n| Final Estimation Result |')
        print('*************************************************************')
        _, final_intent, _, final_slots = parser.parse(text)
        print(f'Final intent: {final_intent}')
        print('Final Slots:')
        for idx, slot in enumerate(final_slots):
            entity = slot['entity']
            raw_value = slot['rawValue']
            print(f'Slot {idx + 1}:')
            print(f'\tentity: {entity}, rawValue: {raw_value}')

        print()
        print('*************************************************************\n')
        if final_intent == 'Introduce':
            print(introduce_dialog)
        elif final_intent == 'HeySnippy':
            print("What's up?")
        elif final_intent == 'GiveMenu':
            print(f"Sorry for your inconvenience, here's the menu\n\n{menu}")

except KeyboardInterrupt:
    print('Program ended')
