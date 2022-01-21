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
import warnings
from copy import copy

import actionlib
import rospy
from home_robot_msgs.msg import IntentManagerAction, IntentManagerGoal, Blacklist, IntentACControllerAction, \
    IntentACControllerGoal, IntentManagerResult, IntentManagerFeedback
from home_robot_msgs.srv import Session, SessionRequest, SessionResponse
from std_srvs.srv import Trigger, TriggerResponse

from core.Dtypes import IntentConfigs
from core.Nodes import Node
from core.tools import Speaker
from core.utils import HeySnipsNLUParser

from rich.console import Console
from rich.traceback import install
from rich.table import Table

console = Console()
print = console.print
install(show_locals=True)


class IntentManager(Node):
    def __init__(self):
        super(IntentManager, self).__init__("intent_manager", anonymous=False)
        with console.status("[yellow] Starting the intent_manager") as status:
            # Load the intent configs
            console.log("Loading configs")
            config_file = rospy.get_param('~config_file')
            self.intent_configs = IntentConfigs(config_file)

            # Load the nlu engines
            console.log("[cyan]Loading NLU engine")
            engine_path = rospy.get_param('~engine_path')
            engine_name = rospy.get_param('~engine_name')
            self.nlu_engine = HeySnipsNLUParser(engine_configs={
                engine_name: engine_path
            })

            # initialize the speaker
            console.log("Initializing speaker")
            self.speaker = Speaker()

            # The Intent blacklist from boss
            self.intent_blacklist = []

            # Get the blacklist from boss
            console.log("[magenta]Getting black list from boss")
            rospy.Subscriber(
                "/intent_boss/blacklist",
                Blacklist,
                self.blacklist_cb,
                queue_size=1
            )

            # Create the session request entry
            console.log("Initializing services")
            rospy.Service('~start_session', Session, self.start_session_cb)
            # Create the continue session entry
            rospy.Service('~continue_session', Session, self.continue_session_cb)
            # Create the stop session entry
            rospy.Service('~stop_session', Trigger, self.stop_session_cb)

            # Calling the SpeechToText node to start flow
            self.s2t_start_session = rospy.ServiceProxy('/voice/start_session', Trigger)
            self.s2t_stop_session = rospy.ServiceProxy('/voice/stop_session', Trigger)

            # Create a instance to call the ActionController
            self.action_controller = actionlib.SimpleActionClient(
                'intent_ac',
                IntentACControllerAction
            )

            # Initialize the action server
            console.log("Starting the action server")
            self.manager_server = actionlib.SimpleActionServer(
                self.name,
                IntentManagerAction,
                execute_cb=self.voice_cb,
                auto_start=False
            )
            self.manager_server.start()

            # session variables
            self.session = None
            rospy.set_param('~on_session', False)

            # The current intent
            self.current_intent = ''

            # continue the session or not
            self.start_session = False
            self.continue_session = False

        print("[bold green]intent_manager is online")
        self.main()

    @staticmethod
    def __on_session():
        return rospy.get_param('~on_session')

    def __intent_in_next_intents(self, intent):
        return intent in self.session.possible_next_intents and len(self.session.possible_next_intents) != 0

    def __establish_session(self, session):
        rospy.set_param('~on_session', True)
        self.session = copy(session)
        try:
            self.s2t_start_session()
        except rospy.ServiceException:
            warnings.warn('Seems like you are debugging the program, Nothing happens', UserWarning)

    def start_session_cb(self, req: SessionRequest):
        if not self.__on_session():
            self.start_session = True
            self.__establish_session(req.session_data)
        return SessionResponse()

    def continue_session_cb(self, req: SessionRequest):
        if self.__on_session():
            self.continue_session = True
            self.__establish_session(req.session_data)
        return SessionResponse()

    def stop_session_cb(self, req):
        if self.__on_session():
            rospy.set_param('~on_session', False)
            self.session = None
            try:
                self.s2t_stop_session()
            except rospy.ServiceException:
                warnings.warn('Seems like you are debugging the program, Nothing happens', UserWarning)
        return TriggerResponse()

    def blacklist_cb(self, blacklist: Blacklist):
        self.intent_blacklist = blacklist.blacklist

    def voice_cb(self, voice_goal: IntentManagerGoal):
        # Get the text from the goal
        raw_text = voice_goal.text
        console.log(f'[bold]Parsing text: "{raw_text}"')

        # Feedback to the voice server that it was successes
        feedback = IntentManagerFeedback()
        feedback.status = 'Accepted'
        self.manager_server.publish_feedback(feedback)

        # Parse the intent from snips-nlu
        result = self.nlu_engine.parse(raw_text)
        intent = result.intent
        slots = result.slots
        self.current_intent = intent

        # Show the report
        console.log(f"Text '{raw_text}' successfully parsed, parsing result:")
        self.__show_nlu_report(intent, slots)

        if intent is None:
            console.log(f"[bold red]Text '{raw_text}' doesn't match any intents")
            intent = 'NotRecognized'

        # Ignore the intent if it was in the blacklist
        if intent in self.intent_blacklist:
            console.log(f'[bold red]Intent {intent} was currently blacklisted by boss, ignoring')
            return

        # Get intent configs from the config file
        try:
            intent_config = self.intent_configs[intent]
        except KeyError:
            console.log(f"[yellow]Intent {intent}'s config doesn't exist, using default")
            intent_config = IntentConfigs.INTENT_DEFAULT_CONFIG

        # Send the intent to the snips_intent_ac
        console.log('Sending intent goal to /intent_ac')

        # Record the intent if it was on session
        if self.__on_session():
            if self.__intent_in_next_intents(intent):
                self.session.flowed_intents.append(intent)
            else:
                self.stop_session_cb(None)
                intent = "NotRecognized"

        intent_goal = IntentACControllerGoal()
        intent_goal.intent = intent
        intent_goal.slots = json.dumps(slots.raw_slots)
        intent_goal.raw_text = raw_text
        if self.__on_session():
            intent_goal.session = self.session

        self.action_controller.send_goal(intent_goal)
        console.log('[bold green]Intent goal sent')

        # Show the preempt info
        allow_preempt = intent_config.allow_preempt
        allow_msg = 'allowed' if allow_preempt else 'not allowed'
        console.log(f"[cyan]Intent {intent} was {allow_msg} to be preempted")

        # If the intent wasn't allowed to be preempted, wait until there's a result
        # else, don't wait
        if not allow_preempt:
            with console.status('[yellow]Waiting for the intent to be executed') as status:
                self.action_controller.wait_for_result()
            print('[bold green]Intent executed successfully')

        result = IntentManagerResult(True)
        self.manager_server.set_succeeded(result, 'The intent was successfully handled')

        if not (self.continue_session or self.start_session):
            self.stop_session_cb(None)

        self.start_session = False
        self.continue_session = False

    @staticmethod
    def __show_nlu_report(intent, slots):
        table = Table()
        print('\n******************\n| Parsing Result |')
        print('*************************************************************')
        print(f'Final intent: {intent}')
        print('Final Slots:')
        table.add_column('Slot idx')
        table.add_column('entity')
        table.add_column('rawValue')
        for idx, slot in enumerate(slots.slots):
            entity = slot.entity
            raw_value = slot.value.rawValue
            table.add_row(str(idx), entity, str(raw_value))

        print(table)
        print('*************************************************************\n')

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = IntentManager()
