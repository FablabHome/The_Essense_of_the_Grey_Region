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
from home_robot_msgs.msg import IntentManagerAction, IntentManagerGoal, Blacklist, IntentACControllerAction, \
    IntentACControllerGoal, IntentManagerResult, IntentManagerFeedback
from home_robot_msgs.srv import StartFlow, StartFlowRequest, StartFlowResponse
from rospkg import RosPack
from std_srvs.srv import Trigger, TriggerResponse

from core.Nodes import Node
from core.utils.KeywordParsers import HeySnipsNLUParser


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

        # Create the flow request entry
        rospy.Service('~start_flow', StartFlow, self.start_flow_cb)
        # Create the stop flow entry
        rospy.Service('~stop_flow', Trigger, self.stop_flow_cb)

        # Calling the SpeechToText node to start flow
        self.s2t_start_flow = rospy.ServiceProxy('/voice/start_flow', Trigger)
        self.s2t_stop_flow = rospy.ServiceProxy('/voice/stop_flow', Trigger)

        # Create a instance to call the ActionController
        self.action_controller = actionlib.SimpleActionClient(
            'snips_intent_ac',
            IntentACControllerAction
        )

        # Initialize the action server
        self.manager_server = actionlib.SimpleActionServer(
            self.name,
            IntentManagerAction,
            execute_cb=self.voice_cb,
            auto_start=False
        )
        self.manager_server.start()

        # The flowing variables
        self.is_flowing = False
        self.flowed_intents = []
        self.possible_next_intents = []

        # The current intent
        self.current_intent = ''

        self.main()

    def start_flow_cb(self, req: StartFlowRequest):
        self.is_flowing = True
        self.flowed_intents.append(self.current_intent)
        self.possible_next_intents = req.next_intents
        self.s2t_start_flow()
        return StartFlowResponse(True)

    def stop_flow_cb(self, req):
        self.is_flowing = False
        self.flowed_intents = []
        self.possible_next_intents = []
        self.s2t_stop_flow()
        return TriggerResponse(success=True, message="The flow has successfully stopped")

    def blacklist_cb(self, blacklist: Blacklist):
        self.intent_blacklist = blacklist.blacklist

    def voice_cb(self, voice_goal: IntentManagerGoal):
        # Get the text from the goal
        raw_text = voice_goal.text
        rospy.loginfo(f'Parsing text: {raw_text}')

        # Feedback to the voice server that it was successes
        feedback = IntentManagerFeedback()
        feedback.status = 'Accepted'
        self.manager_server.publish_feedback(feedback)

        # Parse the intent from snips-nlu
        parse_result = self.nlu_engine.parse(raw_text)

        intent = parse_result.user_intent
        slots = parse_result.parsed_slots

        if intent is None:
            rospy.logwarn(f"Text '{raw_text}' doesn't match any intents")
            intent = 'NotRecognized'

        self.current_intent = intent

        # Show the report
        rospy.loginfo(f"Text '{raw_text}' successfully parsed, parsing result:")
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

        # Send the intent to the snips_intent_a c
        rospy.loginfo('Sending intent to /snips_intent_ac')

        # Record the intent if it was flowing
        if self.is_flowing:
            if intent in self.possible_next_intents:
                self.flowed_intents.append(intent)
            else:
                self.stop_flow_cb(None)

        intent_goal = IntentACControllerGoal()
        intent_goal.intent = intent
        intent_goal.slots = json.dumps(slots)
        intent_goal.raw_text = raw_text
        intent_goal.flowed_intents = self.flowed_intents

        self.action_controller.send_goal(intent_goal)
        rospy.loginfo('Goal sent')

        # Show the preempt info
        allow_preempt = intent_config['allow_preempt']
        allow_msg = 'allowed' if intent_config['allow_preempt'] else 'not allowed'
        rospy.loginfo(f"Intent {intent} was {allow_msg} to be preempted")

        # If the intent wasn't allowed to be preempted, wait until there's a result
        # Else, don't wait
        if not allow_preempt:
            rospy.loginfo('Waiting for the intent to be executed')
            self.action_controller.wait_for_result()
            rospy.loginfo('Intent executed successfully')

        result = IntentManagerResult(True)
        self.manager_server.set_succeeded(result, 'The intent was successfully handled')

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
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = SnipsIntentManager()
