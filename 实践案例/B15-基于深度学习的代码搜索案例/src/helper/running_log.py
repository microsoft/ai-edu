# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import datetime
import sys
import os
import atexit

from .file import load_json, save_json


class RunningLog:
    def __init__(self, log_path):
        self.running = {
            'start': str(datetime.datetime.now()),
            'end': None,
            'argv': sys.argv,
            'parameters': {},
            'state': 'failed'
        }

        def save_running_log():
            print('saving running log to running-log.json')
            self.running['end'] = str(datetime.datetime.now())
            filename = os.path.join(log_path, 'running-log.json')
            all_running = []
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    all_running = load_json(filename)
            all_running.append(self.running)
            save_json(all_running, filename)
        atexit.register(save_running_log)

    def set(self, key, value):
        self.running[key] = value

