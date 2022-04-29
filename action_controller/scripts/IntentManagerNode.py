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
import hashlib
import json
from copy import copy
from datetime import datetime

import actionlib
import rospy
from home_robot_msgs.msg import IntentManagerAction, IntentManagerGoal, Blacklist, IntentACControllerAction, \
    IntentACControllerGoal, IntentManagerResult, IntentManagerFeedback
from home_robot_msgs.srv import Session, SessionRequest, SessionResponse
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.traceback import install
from std_srvs.srv import Trigger, TriggerResponse

from core.Dtypes import IntentConfigs
from core.Nodes import Controller
from core.utils import HeySnipsNLUParser

console = Console()
print = console.print
install(show_locals=True)


class IntentManager(Controller):
    def __init__(self):
        super(IntentManager, self).__init__(
            name="intent_manager",
            anonymous=False,
            default_state='NORMAL',
            state_param_group='/intent/',
            states=['NORMAL', 'ON_SESSION', 'SLOT_MISSING']
        )
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

        print("[bold green]intent_manager is online")
        self.main()

    """
    ********************************* Session callbacks starts here ***********************************
    """
    def start_session_cb(self, req: SessionRequest):
        if not self.__on_session():
            # Generate session ID
            req.session_data.id = self.__generate_session_id()
            console.log(f'[bold yellow]A session has been established with the following parameters:')
            self.__show_session_summary(req.session_data)
            self.__establish_session(req.session_data)
        return SessionResponse()

    def continue_session_cb(self, req: SessionRequest):
        if self.__on_session():
            console.log(f'[bold yellow]The session has been continued with the following parameters:')
            self.__show_session_summary(req.session_data)
            self.__establish_session(req.session_data)
        return SessionResponse()

    def stop_session_cb(self, req):
        if self.__on_session():
            rospy.set_param('~on_session', False)
            self.session = None
        return TriggerResponse()
    """
    ********************************* Session callbacks ends here *************************************
    """

    def blacklist_cb(self, blacklist: Blacklist):
        self.intent_blacklist = blacklist.blacklist

    """
    ///////////////////////////////// Helper functions starts here /////////////////////////////////////
    """
    @staticmethod
    def __on_session():
        return rospy.get_param('~on_session')

    def __intent_in_next_intents(self, intent):
        return (intent in self.session.possible_next_intents) or len(self.session.possible_next_intents) == 0

    def __establish_session(self, session):
        rospy.set_param('~on_session', True)
        self.session = copy(session)

    @staticmethod
    def __show_session_summary(session):
        summary = Table(show_lines=True)
        summary.add_column('Parameter')
        summary.add_column('Value')
        summary.add_row('session_id', session.id)
        summary.add_row('started_intent', session.started_intent)
        summary.add_row('custom_data', session.custom_data)
        summary.add_row('max_rounds', Text(str(session.max_rounds), style='bold magenta'))
        summary.add_row('elapsed_rounds', str(session.elapsed_rounds))
        print(summary)

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

    @staticmethod
    def __generate_session_id():
        date = str(datetime.now())
        sess_id = hashlib.sha256(date.encode())
        return sess_id.hexdigest()

    def __handle_feedback(self):
        feedback = IntentManagerFeedback()
        feedback.status = 'Accepted'
        self.manager_server.publish_feedback(feedback)

    def __check_if_end_session(self):
        # If no session is going to establish, end the session
        if self.__on_session():
            if self.session.elapsed_rounds >= self.session.max_rounds:
                self.stop_session_cb(None)

    """
    ///////////////////////////////// Helper functions ends here /////////////////////////////////////
    """

    def voice_cb(self, voice_goal: IntentManagerGoal):
        # Set the state to NORMAL first, if there isn't any request, it will stay the same
        # self.set_state('NORMAL')

        # Get the text from the goal
        raw_text = voice_goal.text
        console.log(f'[bold]Parsing text: "{raw_text}"')

        # Feedback to the voice server that it was successes
        self.__handle_feedback()

        # Parse the intent from snips-nlu
        result = self.nlu_engine.parse(raw_text)
        intent = result.intent
        slots = result.slots

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
            self.session.elapsed_rounds += 1
            if self.__intent_in_next_intents(intent):
                self.session.flowed_intents.append(intent)
            else:
                self.stop_session_cb(None)
                intent = "NotRecognized"

        # Serialize information into ROS message
        intent_goal = IntentACControllerGoal()
        intent_goal.intent = intent
        intent_goal.slots = json.dumps(slots.raw_slots)
        intent_goal.raw_text = raw_text
        if self.__on_session():
            intent_goal.session = self.session

        self.action_controller.send_goal(intent_goal)
        console.log('[bold green]Intent goal sent')

        # Show preempt info
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

        # If no session is going to establish, end the session
        self.__check_if_end_session()

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = IntentManager()
