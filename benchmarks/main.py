import synthesis
import examples
import random

global USE_OPTIONALS
USE_OPTIONALS = True

print("doin' VSA stuff")
for i, ex in enumerate(examples.all_benchmarks):
    print(f'Example {i}:')
    regex = synthesis.synthesize(ex.inputs[:3])
    print(f'  synthesized: {regex}')
    print(f'  correct: {ex.regex}')





