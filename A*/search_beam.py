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

class Bucket(list):
    '''Yes that's right it's a bucket!!!
    Keep the ones with small score. Uses a max heap internally
    '''

    max_size: int

    def __init__(self, max_size: int):
        self.max_size = max_size

    def add(self, elt: State):
        if len(self) == self.max_size:
            if elt.score >= self[0].score: return
            self[0] = elt
            self.bubble_down(0)
        else:
            self.append(elt)
            if len(self) == self.max_size: self.heapify()

    def bubble_down(self, i: int):
        x = self[i]
        while True:
            l = 2*i + 1
            r = 2*i + 2
            if l >= len(self): break
            child = r if r < len(self) and self[r].score > self[l].score else l
            if self[child].score > x.score:
                self[i] = self[child]
                i = child
            else:
                break
        self[i] = x

    def heapify(self):
        for i in reversed(range((len(self)+1)//2)):
            self.bubble_down(i)

def search_top_k(examples: List[str], beam_size: int) -> State:
    N = len(examples)
    sum_lengths = sum(len(ex) for ex in examples)

    heuristic = TwoTotalHeuristic(examples) # TwoMaxHeuristic(examples)
    VSAs = [VSA.single_example(ex) for ex in examples]

    beams = {i: Bucket(beam_size) for i in range(sum_lengths+1)}
    beams[0].add(State(
        indices = (0,) * N,
        score_so_far = 0.,
        score = heuristic.value_at((0,) * N),
        regex_so_far = []
    ))
    for i in range(sum_lengths):
        for state in beams[i]:
            for end, r in next_states(VSAs, state.indices):
                score_so_far = state.score_so_far \
                    + r.simplicity_score() \
                    + sum(r.specificity_score(e[a:b]) for e, a, b in zip(examples, state.indices, end))
                beams[sum(end)].add(State(
                        indices = end,
                        score_so_far = score_so_far,
                        score = score_so_far + heuristic.value_at(end),
                        regex_so_far = state.regex_so_far + [r]))

    # Find the best in the last bucket
    # return min((s.score, s) for s in beams[sum_lengths])[1]
    return [s for s in beams[sum_lengths]] # HACK

def search(examples: List[str], beam_size: int) -> State:
    return min(search_top_k(examples, beam_size), key=lambda s: s.score)

def next_states(VSAs: List[VSA], starting: VSAState) -> Iterator[Tuple[VSAState, Regex]]:
    if len(VSAs) == 1:
        for end, rs in VSAs[0].edges[starting].items():
            for r in rs:
                yield end, r
        return
    v, *rest_vsas = VSAs
    i, *rest_starting = starting
    rest_starting = tuple(rest_starting)
    for rest_end, r in next_states(rest_vsas, rest_starting):
        yield (i, *rest_end), r.opt()
        for e, rs in v.edges[(i,)].items():
            if r in rs or (isinstance(r, Optional) and r.contents in rs):
                yield e + rest_end, r
    for e, rs in v.edges[(i,)].items():
        end = e + rest_starting
        for r in rs:
            yield end, r.opt()

if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)
    result = search(inputs, 10)
    print(result.score, ''.join(str(r) for r in result.regex_so_far))
