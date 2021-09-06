#!/usr/bin/env python3
import argparse
import rospy
import sys

from mr_voice.srv import SpeakerSrv
from std_msgs.msg import String


def main(args):
    global pub, service
    if args['wait_until_end']:
        service(args['text'])
    else:
        pub.publish(args['text'])


if __name__ == '__main__':
    rospy.init_node('say', anonymous=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--text', type=str, required=True, help='Text you want the computer to say')
    parser.add_argument('-r', '--rate', type=int, default=130, help='Speed of the speaker')
    parser.add_argument('-v', '--volume', type=float, default=1.0, help='Volume of the speaker')
    parser.add_argument('-l', '--language', type=str, default='en-us', help='Language of the speaker')
    parser.add_argument('-w', '--wait-until-end', action='store_true')

    args = vars(parser.parse_args())

    pub = rospy.Publisher(
        '/speaker/say',
        String,
        queue_size=1
    )
    service = rospy.ServiceProxy('/speaker/text', SpeakerSrv)
    rospy.sleep(.5)
    try:
        main(args)
        sys.exit(0)
    except Exception as e:
        print(f'Program ended due to: {e}')
        sys.exit(1)
