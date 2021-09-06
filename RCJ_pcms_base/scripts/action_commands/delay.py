#!/usr/bin/env python3

import time
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seconds', type=int,
                        required=True, help='Seconds to delay')
    args = vars(parser.parse_args())
    try:
        time.sleep(args['seconds'])
        sys.exit(0)
    except Exception as e:
        print(f'Program ended due to {e}')
        sys.exit(1)
