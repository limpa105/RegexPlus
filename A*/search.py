from typing import *
import math, itertools, functools, dataclasses, heapq
from Heuristics import *
import os
from VSA import *
from  PriorityQueue import *
from Regex import *

@dataclasses.dataclass(frozen=True)
class State:
    '''The thing that goes into a priority queue'''
    indices: Tuple[int, ...]
    score_so_far: float
    score: float  # Adding score_so_far + heuristic of score to go
    regex_so_far: List[Regex]

    def __eq__(self, other) -> bool:
        return self.score == other.score

### The main search code

def search(examples: List[str], max_size: int) -> State:
    N = len(examples)
    pq = LimitedPQ(max_size)
    heuristic = TwoMaxHeuristic(examples)
    print('computed heuristics')

    starting_index = (0,) * N
    ending_index = tuple(len(e) for e in examples)
    pq.add(State(
        indices=starting_index,
        score_so_far=0.,
        score=heuristic.value_at(starting_index),
        regex_so_far=[]
    ))
    VSAs = [VSA.single_example(ex) for ex in examples]

    while len(pq) > 0:
        state = pq.pop_best()
        if state.indices == ending_index:
            return state
        for end, r in next_states(VSAs, state.indices):
            score_so_far = state.score_so_far \
                + r.simplicity_score() \
                + sum(r.specificity_score(e[a:b]) for e, a, b in zip(examples, state.indices, end))
            pq.add(State(indices=end,
                    score_so_far = score_so_far,
                    score = score_so_far + heuristic.value_at(end),
                    regex_so_far = state.regex_so_far + [r]))
    raise Exception('No regex works! (Should never happen)')


def next_states(VSAs: list[VSA], starting: VSAState):
    if len(VSAs) == 1:
        return [(end, r) for end, rs in VSAs[0].edges[starting].items() for r in rs]
    v, *rest_vsas = VSAs
    i, *rest_starting = starting
    rest_starting = tuple(rest_starting)
    rest = next_states(rest_vsas, rest_starting)
    return [(e + rest_end, r) for rest_end, r in rest for e, rs in v.edges[(i,)].items() if r in rs] \
        + [((i, *rest_end), r.opt()) for rest_end, r in rest] \
        + [(e + rest_starting, r.opt()) for e, rs in v.edges[(i,)].items() for r in rs]


if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)
    result = search(inputs, 10000)
    print(result.score, ''.join(str(r) for r in result.regex_so_far))
