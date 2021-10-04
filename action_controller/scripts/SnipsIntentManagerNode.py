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
from os import path

import actionlib
import rospy
from rospkg import RosPack

from core.Nodes import Node
from core.utils.KeywordParsers import HeySnipsNLUParser
from home_robot_msgs.msg import IntentManagerAction, IntentManagerGoal, Blacklist


class SnipsIntentManager(Node):
    INTENT_DEFAULT_CONFIG = {
        'allow_preempt': False
    }

    def __init__(self):
        super(SnipsIntentManager, self).__init__("snips_intent_manager", anonymous=False)

        # Base path of the configs
        base = RosPack().get_path('rcj_pcms_base') + '/../config/ActionController/SnipsIntentConfigs'
        # Load the intent configs
        config_file = rospy.get_param('~config_file')
        with open(path.join(base, config_file)) as config:
            self.intent_configs = json.load(config)

        # Load the nlu engines
        engine_path = rospy.get_param('~engine_path')
        engine_name = rospy.get_param('~engine_name')
        self.nlu_engine = HeySnipsNLUParser(engine_configs={
            engine_name: engine_path
        })

        # The Intent blacklist from boss
        self.intent_blacklist = []
        
        # Get the blacklist from boss
        rospy.Subscriber(
            "/snips_intent_boss/blacklist",
            Blacklist,
            self.blacklist_cb,
            queue_size=1
        )

        # Initialize the action server
        self.manager_server = actionlib.SimpleActionServer(
            self.name,
            IntentManagerAction,
            execute_cb=self.voice_cb,
            auto_start=False
        )
        self.manager_server.start()

    def blacklist_cb(self, blacklist: Blacklist):
        self.intent_blacklist = blacklist.blacklist

    def voice_cb(self, voice_goal: IntentManagerGoal):
        # Get the text from the goal
        text = voice_goal.text
        rospy.loginfo(f'Parsing text: {text}')
        # Parse the intent from snips-nlu
        parse_result = self.nlu_engine.parse(text)
        
        intent = parse_result.user_intent
        slots = parse_result.parsed_slots
        
        if intent is None:
            rospy.logwarn(f"Text '{text}' doesn't match any intents")
            return
        # Show the report
        rospy.loginfo(f"Text '{text}' successfully parsed, parsing result:")
        self.__show_nlu_report(intent, slots)

        # Ignore the intent if it was in the blacklist
        if intent in self.intent_blacklist:
            rospy.logerr(f'Intent {intent} was currently blacklisted by boss, ignoring')
            return

        # Get intent configs from the config file
        try:
            intent_config = self.intent_configs[intent]
        except KeyError:
            rospy.logwarn(f"Intent {intent}'s config doesn't exist, using default")
            intent_config = SnipsIntentManager.INTENT_DEFAULT_CONFIG

        allow = 'allowed' if intent_config['allow_preempt'] else 'not allowed'
        rospy.loginfo(f"Intent {intent} was {allow} to be preempted")
        
    @staticmethod
    def __show_nlu_report(intent, slots):
        print('\n***************************\n| Parsing Result |')
        print('*************************************************************')
        print(f'Final intent: {intent}')
        print('Final Slots:')
        for idx, slot in enumerate(slots):
            entity = slot['entity']
            raw_value = slot['rawValue']
            print(f'Slot {idx + 1}:')
            print(f'\tentity: {entity}, rawValue: {raw_value}')

        print()
        print('*************************************************************\n')

    def main(self):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    node = SnipsIntentManager()
