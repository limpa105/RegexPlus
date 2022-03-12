import synthesis
import examples
import random

global USE_OPTIONALS
USE_OPTIONALS = True

NUM_EXAMPLES = 4

print("doin' VSA stuff")
for i, ex in enumerate(examples.all_benchmarks):
    print(f'Example {i}:')
    print(ex.inputs[:NUM_EXAMPLES])
    print(f'  correct: {ex.regex}')
    print('  synthesized:')
    for wt, regex in synthesis.synthesize(ex.inputs[:NUM_EXAMPLES]):
        print('   - [%.3f] %s' % (wt, regex))

