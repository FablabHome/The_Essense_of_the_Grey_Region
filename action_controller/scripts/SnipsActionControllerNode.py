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

from core.Nodes import ActionEvaluator


class SnipsActionControllerNode(ActionEvaluator):
    def __init__(self):
        # Initialize the intent to callback map, must have NotRecognized situation
        self.intent2callback = {
            'turnLightOn': self.__turnlightson,
            'setLightColor': self.__setcolor,
            'NotRecognized': self.__notrecognized
        }

        super(SnipsActionControllerNode, self).__init__()

    def __turnlightson(self, intent, slots, raw_text, flowed_intents):
        room = None
        if len(slots) == 0:
            self.speaker.say_until_end("OK, but which room?")
            self.start_flow(next_intents=['turnLightOn'])
            return

        if flowed_intents[len(flowed_intents) - 1] == 'turnLightOn':
            room = slots[0]['value']['value']

        self.speaker.say_until_end(f"Ok, let's light up the {room}, you can set the color or the intensity")
        self.start_flow(next_intents=['setLightColor'])

    def __setcolor(self, intent, slots, raw_text, flowed_intents):
        if len(flowed_intents) == 0:
            self.speaker.say_until_end("Ok settings the lights color")

        elif flowed_intents[0] == 'turnLightOn':
            self.speaker.say_until_end("Ok, lets change color to " + slots[0]['value']['value'])
            self.stop_flow()

    def __notrecognized(self, intent, slots, raw_text, flowed_intents):
        if len(flowed_intents) == 0:
            self.speaker.say_until_end("Sorry, I don't understand that")
        elif flowed_intents[0] == 'turnLightsOn':
            self.speaker.say_until_end("Sorry, as you don't want to do more")

    def main(self):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    node = SnipsActionControllerNode()
