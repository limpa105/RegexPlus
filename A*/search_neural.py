from listener import *
from model import * 
from typing import *
import math, itertools, functools, dataclasses, heapq
from Heuristics import *
import os
from VSA import *
from  PriorityQueue import *
from Regex import *
from NN import NNState

NUM_OUTGOING_EDGES = 15

@dataclasses.dataclass(frozen=True)
class State:
    '''The thing that goes into a priority queue'''
    indices: Tuple[int, ...]
    score_so_far: float
    score: float  # Adding score_so_far + heuristic of score to go
    regex_so_far: List[Regex]
    nn_state: NNState

def search(examples: List[str], max_size: int) -> State:
    N = len(examples)
    pq = LimitedPQ(max_size)
    heuristic = TwoMaxHeuristic(examples)
    # print('computed heuristics')

    starting_index = (0,) * N
    ending_index = tuple(len(e) for e in examples)
    pq.add(State(
        indices=starting_index,
        score_so_far=0.,
        score=heuristic.value_at(starting_index),
        regex_so_far=[],
        nn_state=NNState.initial(examples)
    ))
    VSAs = [VSA.single_example(ex) for ex in examples]

    while len(pq) > 0:
        state = pq.pop_best()
        if state.indices == ending_index:
            return state
        for end, nn_state, r in next_states(examples, state):
            score_so_far = state.score_so_far \
                + r.simplicity_score() \
                + sum(r.specificity_score(e[a:b]) for e, a, b in zip(examples, state.indices, end))
            pq.add(State(indices=end,
                    score_so_far = score_so_far,
                    score = score_so_far + heuristic.value_at(end),
                    regex_so_far = state.regex_so_far + [r],
                    nn_state = nn_state))
    raise Exception("No regex works! (This is the neural network's fault")

def next_states(examples: List[str], state: State) -> Iterator[Tuple[VSAState, NNState, Regex]]:
    for r, nn_state in state.nn_state.best_next_edges(NUM_OUTGOING_EDGES):
        for end in itertools.product(*(r.next_inds(ex, i) for ex, i in zip(examples, state.indices))):
            yield end, nn_state, r

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
