import re
import itertools
from typing import *
from somedata import all_regexes, all_inputs, benchmarks

# all_regexes: List of regexes
# all_inputs : List of inputs

def synthesis(inputs: List[str]) -> str:
    # The inputs all come from the list of all_inputs
    # Try to pick the best matching regex from the list of all_regexes
    assert all(i in all_inputs for i in inputs)

    # Give the top 5 possibilities
    possibilities = [(regex, L1(regex, inputs)) for regex in all_regexes if matches(regex, inputs)]
    possibilities.sort(key = lambda p: -p[1])
    return possibilities[:5]

def L1(regex: str, inputs: List[str]) -> float:
    return P_R(regex) * S1(regex, inputs)

def S1_numerator(regex: str, inputs: List[str]) -> float:
    # total_prob = 1
    # for i in inputs:
    #     total_prob = total_prob * P_W(i)
    return L0(regex, inputs) * P_W(inputs)

def S1_helper(regex: str, fixed_inputs: List[str], new_input: str) -> float:
    return (
        S1_numerator(regex, fixed_inputs + [new_input])
        /
        sum(S1_numerator(regex, fixed_inputs + [w]) for w in all_inputs)
    )

def S1(regex, inputs):
    P = 1.0
    for i in range(len(inputs)):
        P *= S1_helper(regex, inputs[:i], inputs[i])
    return P

def L0(regex: str, inputs: List[str]) -> float:
    # probability of regex over all regexes that match
    if not matches(regex, inputs): return 0
    return (P_R(regex)) / sum(P_R(r) for r in all_regexes if matches(r, inputs))


match_table = {(r,w) for r in all_regexes for w in all_inputs if re.fullmatch(r,w)}
def matches(regex: str, inputs: List[str]) -> bool:
    return all((regex,i) in match_table for i in inputs)

# Priors: they don't have to add up to 1 since we're gonna normalize everything anyways

# TODO
# currently just using length of rgex
def P_R(r: str) -> float:
    return 1/(25) ** (len(r) + 1)

#
def P_W(inputs: List[str]) -> float:
    return 1/96**sum(len(w)+1 for w in inputs)

if __name__ == '__main__':
    for i,k in enumerate(benchmarks):
        print(f'Benchmark number {i+1}: {k}')
        result = synthesis(k)
        for r, score in result:
            print(' - [%.2e] %s' %(score, r))

