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
import json

import actionlib
import rospy
from home_robot_msgs.msg import IntentACControllerAction, IntentACControllerResult, IntentACControllerGoal

from core.Nodes import Node


def dummy_callback(intent, slots, raw_text, flowed_intents):
    return


class SnipsActionControllerNode(Node):
    def __init__(self):
        super(SnipsActionControllerNode, self).__init__("snips_intent_ac", anonymous=False)

        # Initialize the intent to callback map, must have NotRecognized situation
        self.intent2callback = {
            'Introduce': self.__introduce,
            'GiveMenu': self.__show_menu,
            'OrderFood': self.__order_food,
            'OrderFoodTakeOut': self.__order_food,
            'NotRecognized': self.__not_recognized
        }

        # Initialize the action server
        self.action_controller_server = actionlib.SimpleActionServer(
            self.name,
            IntentACControllerAction,
            execute_cb=self.message_cb,
            auto_start=False
        )
        self.action_controller_server.start()
        self.main()

    def message_cb(self, goal: IntentACControllerGoal):
        # TODO async the callback function for preempt checking
        # Parse the data
        intent = goal.intent
        slots = json.loads(goal.slots)
        raw_text = goal.raw_text
        flowed_intents = goal.flowed_intents

        # Execute the callbacks
        if intent not in self.intent2callback:
            rospy.logerr(f"Intent {intent}'s callback doesn't exist, doing nothing")
            callback = dummy_callback
        else:
            callback = self.intent2callback[intent]

        callback(intent, slots, raw_text, flowed_intents)

        # Set the callback was executed successfully
        result = IntentACControllerResult(True)
        self.action_controller_server.set_succeeded(result)

    @staticmethod
    def __introduce(intent, slots, raw_text, flowed_intents):
        introduce_dialog = '''
        Ah, Forgive me for not introducing myself, masters.
        I'm snippy, your virtual assistant in this restaurant,
        I'm still under development, so you could only see me talking
        right now.
        '''
        print(introduce_dialog)

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

    @staticmethod
    def __order_food(intent, slots, raw_text, flowed_intents):
        # amounts = []
        # orders = []
        # message = 'User ordered: '
        # orders_message = []
        # to_go = ''
        # for idx, slot in enumerate(slots):
        #     amounts.append(1)
        #     if slot['slotName'] == 'amount':
        #         amounts[idx] = int(slot['value']['value'])
        #     if slot['slotName'] == 'food':
        #         orders.append(slot['value']['value'])
        #     if slot['slotName'] == 'takeout':
        #         to_go = 'to go'
        #     orders_message.append(f'{amounts[idx]} {orders[idx]}')
        #
        # orders_message.append(to_go)
        # message += ', '.join(orders_message)
        # rospy.loginfo(message)
        rospy.loginfo(slots)

    @staticmethod
    def __not_recognized(intent, slots, raw_text, flowed_intents):
        rospy.loginfo(f"Currently there isn't an action for '{raw_text}'")

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = SnipsActionControllerNode()
