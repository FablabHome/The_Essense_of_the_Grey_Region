#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2020 rootadminWalker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import rospy

from core.Nodes import ActionEvaluator


def dummy_callback(intent, slots, raw_text, flowed_intents):
    return


class SnipsActionControllerNode(ActionEvaluator):
    def __init__(self):
        # Initialize the intent to callback map, must have NotRecognized situation
        self.intent2callback = {
            'Introduce': self.__introduce,
            'GiveMenu': self.__show_menu,
            'OrderFood': self.__order_food,
            'OrderFoodTakeOut': self.__order_food,
            'NotRecognized': self.__not_recognized
        }

        super(SnipsActionControllerNode, self).__init__()

    def __introduce(self, intent, slots, raw_text, flowed_intents):
        introduce_dialog = '''
        Ah, Forgive me for not introducing myself, masters.
        I'm snippy, your virtual assistant in this restaurant,
        I'm still under development, so you could only see me talking
        right now.
        '''
        self.speaker.say_until_end(introduce_dialog)

    @staticmethod
    def __show_menu(intent, slots, raw_text, flowed_intents):
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
        hot dog                        $10
        '''
        print(f"Sorry for your inconvenience, here's the menu\n\n{menu}")

    def __order_food(self, intent, slots, raw_text, flowed_intents):
        order_what = False
        orders = {}
        i = 0
        while i < len(slots):
            if slots[i]['slotName'] == 'amount':
                amount = int(slots[i]['value']['value'])
                try:
                    next_slot = slots[i + 1]
                    if next_slot['slotName'] == 'food':
                        orders[next_slot['value']['value']] = amount
                        i += 2
                    elif next_slot['slotName'] == 'amount':
                        orders[f'Unknown{i}'] = amount
                        i += 1
                        order_what = True

                except IndexError:
                    order_what = True
                    orders[f'Unknown{i}'] = amount
                    i += 1
            elif slots[i]['slotName'] == 'food':
                orders[slots[i]['value']['value']] = 1
                i += 1

        if order_what or len(slots) == 0:
            self.speaker.say_until_end("I'm sorry, but could you repeat it again?")
            self.start_flow(next_intents=['OrderFood', 'NotRecognized'])
            return

        if len(flowed_intents) > 0:
            if set(flowed_intents) == {'OrderFood'}:
                if not order_what:
                    self.stop_flow()

        self.speaker.say_until_end('Ok, Gotcha')
        print(orders)

    def __not_recognized(self, intent, slots, raw_text, flowed_intents):
        if len(flowed_intents) == 0:
            rospy.loginfo(f"Currently there isn't an action for '{raw_text}'")
        elif flowed_intents[0] == 'OrderFood':
            rospy.loginfo('Sorry, I could not understand what do you want to order, please say it again')
            self.stop_flow()

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = SnipsActionControllerNode()
