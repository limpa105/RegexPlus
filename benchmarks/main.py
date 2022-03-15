
import synthesis
import examples
import random
import os
import signal
import sys
import time
import gc

global USE_OPTIONALS
USE_OPTIONALS = True

NUM_EXAMPLES = 4
TIMEOUT_SEC = 40

if sys.argv:
    NUM_EXAMPLES = int(sys.argv[1])
if len(sys.argv) > 2:
    TIMEOUT_SEC = int(sys.argv[2])

print(NUM_EXAMPLES)
print(TIMEOUT_SEC)

# Using code from https://erogol.com/timeout-function-takes-long-finish-python/
class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError()
        gc.collect()
        print("Hello?")
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

print("doin' VSA stuff")
for i, ex in enumerate(examples.all_benchmarks):
    print(f'Example {i}:')
    print(ex.inputs[:NUM_EXAMPLES])
    print(f'  correct: {ex.regex}')
    print('  synthesized:')
    with timeout(seconds=TIMEOUT_SEC):
        try:
    	    for wt, regex in synthesis.synthesize(ex.inputs[:NUM_EXAMPLES]):
                print('   - [%.3f] %s' % (wt, regex))
        except TimeoutError:
            print("Timed out")
            pass

