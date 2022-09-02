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
import os
from copy import copy
from datetime import datetime

import actionlib
import rospy
from home_robot_msgs.msg import IntentManagerAction, IntentManagerGoal, Blacklist, IntentACControllerAction, \
    IntentACControllerGoal, IntentManagerResult, VoiceSession
from home_robot_msgs.srv import Session, SessionRequest, SessionResponse, ContinueSess, ContinueSessResponse, \
    ContinueSessRequest
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.traceback import install
from std_srvs.srv import Trigger, TriggerResponse

from core.Dtypes import IntentConfigs, Slots
from core.Nodes import Node
from core.utils import HeySnipsNLUParser

console = Console()
print = console.print
install(show_locals=True)


class IntentManager(Node):
    def __init__(self):
        super(IntentManager, self).__init__(
            name="intent_manager",
            anonymous=False,
        )
        with console.status("[yellow] Starting the intent_manager") as _:
            # Load the intent configs
            console.log("Loading configs")
            config_file = rospy.get_param('~config_file')
            self.intent_configs = IntentConfigs(config_file)

            # Load the nlu engines
            console.log("[cyan]Loading NLU engine")
            engine_configs = {}
            dataset_configs = {}
            for engine_path in self.intent_configs.engines:
                engine_name = engine_path.split(os.path.sep)[-1]
                engine_configs[engine_name] = engine_path

            for dataset_path in self.intent_configs.datasets:
                dataset_name = dataset_path.split(os.path.sep)[-1].split('.')[0]
                dataset_configs[dataset_name] = dataset_path

            self.nlu_engine = HeySnipsNLUParser(engine_configs=engine_configs)

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

            rospy.Service('~start_session', Session, self.start_session_cb)
            rospy.Service('~continue_session', ContinueSess, self.continue_session_cb)
            rospy.Service('~stop_session', Trigger, self.stop_session_cb)

            # Create an instance to call the ActionController
            self.action_evaluator = actionlib.SimpleActionClient(
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
            self.session = VoiceSession()

        print("[bold green]intent_manager is online")
        self.main()

    def blacklist_cb(self, blacklist):
        pass

    def get_current_resp(self, current_intent, missing_slots):
        confirm_responses = self.intent_configs.intents[current_intent].confirm_responses
        missing_slots = set(missing_slots)
        for resp_key, confirm_resp in confirm_responses.__dict__.items():
            if set(resp_key.split()) == missing_slots:
                return confirm_resp

    def start_session_cb(self, session: SessionRequest):
        self.session = copy(session.session_data)
        self.session.id = self.__generate_session_id()
        self.session.session_type = 'NORMAL'
        self.session.max_rounds = 1
        self.session.elapsed_rounds = 0
        return SessionResponse()

    def continue_session_cb(self, continue_req: ContinueSessRequest):
        self.session.custom_data = continue_req.custom_data
        self.session.possible_next_intents = continue_req.possible_next_intents
        self.session.max_rounds += 1
        return ContinueSessResponse()

    def start_confirm_session(self, current_intent, slots, max_re_ask, missing_slots):
        assert not self.on_session(), f"A session with\nID: {self.session.id}\ntype:{self.session.session_type}\nis on"

        pass_data = {'confirm_resp': self.get_current_resp(current_intent, missing_slots), 'slots': slots.raw_slots}
        self.session = VoiceSession()
        self.session.id = self.__generate_session_id()
        self.session.started_intent = current_intent
        self.session.possible_next_intents = [current_intent]
        if (ci := self.intent_configs.intents[current_intent].confirm_intent) != '':
            self.session.possible_next_intents.append(ci)
        self.session.custom_data = json.dumps(pass_data)
        self.session.session_type = 'CONFIRM'
        self.session.max_rounds = max_re_ask
        self.session.elapsed_rounds = 0

    def update_confirm_session(self, this_time_slots, missing_slots):
        assert self.session.session_type == 'CONFIRM', "Current session running is not a confirm session"
        pass_data = json.loads(self.session.custom_data)
        saved_slots = pass_data['slots']
        saved_slots = Slots(saved_slots)
        saved_slots.update_but_ignore_exist_slots(this_time_slots.raw_slots)
        pass_data['confirm_resp'] = self.get_current_resp(self.session.started_intent, missing_slots)
        pass_data['slots'] = saved_slots.raw_slots
        self.session.custom_data = json.dumps(pass_data)

    def stop_session_cb(self, req):
        self.session = VoiceSession()
        return TriggerResponse()

    def on_session(self):
        return self.session.id != ''

    @staticmethod
    def __show_session_summary(session):
        summary = Table(show_lines=True)
        summary.add_column('Parameter')
        summary.add_column('Value')
        summary.add_row('session_id', session.id)
        summary.add_row('started_intent', session.started_intent)
        summary.add_row('session_type', session.session_type)
        summary.add_row('custom_data', session.custom_data)
        summary.add_row('flowed_intents', '-->'.join(session.flowed_intents))
        summary.add_row('possible_next_intents', ', '.join(session.possible_next_intents))
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

    def parse_data(self, text):
        _, intent, _, slots, = self.nlu_engine.parse(text)[-1]
        slots = Slots(slots)
        if intent is None:
            intent = 'NotRecognized'

        missing_slots = self.intent_configs.find_missing_slots(intent, slots)
        if self.on_session():
            if self.session.session_type == 'NORMAL':
                if intent not in self.session.possible_next_intents:
                    intent = 'NotRecognized'
            elif self.session.session_type == 'CONFIRM':
                missing_slots = set(missing_slots) - set(
                    Slots(json.loads(self.session.custom_data)['slots']).list_slot_names())
                if len(missing_slots) > 0:
                    self.update_confirm_session(slots, missing_slots)
                    intent = 'Confirm'
                else:
                    self.stop_session_cb(None)
        else:
            if len(missing_slots) > 0:
                if self.session.session_type != 'CONFIRM':
                    self.start_confirm_session(intent, slots, self.intent_configs.intents[intent].max_re_ask,
                                               missing_slots)
                intent = 'Confirm'

        return intent, slots

    def voice_cb(self, voice_goal: IntentManagerGoal):
        text = voice_goal.text
        send_intent, slots = self.parse_data(text)

        if self.on_session():
            if self.session.elapsed_rounds >= self.session.max_rounds:
                if self.session.session_type == 'CONFIRM':
                    send_intent = 'NotRecognized'
                self.stop_session_cb(None)
            else:
                console.log("This is a session running currently:")
                self.__show_session_summary(self.session)
                self.session.elapsed_rounds += 1

        self.__show_nlu_report(send_intent, slots)

        goal = IntentACControllerGoal()
        goal.intent = send_intent
        goal.slots = json.dumps(slots.raw_slots)
        goal.raw_text = text
        goal.session = self.session

        self.action_evaluator.send_goal(goal)
        self.action_evaluator.wait_for_result()
        self.manager_server.set_succeeded(IntentManagerResult(self.on_session()))

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = IntentManager()
