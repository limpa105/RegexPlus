from typing import *
import math, itertools, functools, dataclasses, heapq
from Regex import *
from Heuristics import *
import os

@functools.total_ordering
@dataclasses.dataclass(frozen=True)
class State:
    '''The thing that goes into a priority queue'''
    indices: Tuple[int, ...]
    score_so_far: float
    score: float  # Adding score_so_far + heuristic of score to go
    regex_so_far: List[Regex]

    def __eq__(self, other) -> bool:
        return self.score == other.score
    def __lt__(self, other) -> bool:
        return self.score < other.score


class PQ(Protocol):
    '''
    The protocol that is implemented by:
     - PriorityQueue
     - Amortized Limited memory priority queue
     - MinMax heap limited memory priority queue
    '''
    def add(self, elt: State):
        ...
    def pop_best(self) -> State:
        ...

class LimitedPQ(list[State]):
    '''
    A mutable priority queue, that you add stuff to over time, with limited space
    '''

    def __init__(self: 'LimitedPQ', max_size: int):
        self.max_size = max_size

    def kill_extra_elts(self: 'Shelf'):
        '''quickselect then re-heapify: O(n)'''
        if len(self) <= self.max_size: return

        lo = 0
        while len(self) > self.max_size:
            cut_elt = self[self.max_size]
            mid = partition(self, lo, len(self), lambda x: x.score < cut_elt.score)
            if mid >= self.max_size:
                while len(self) > mid: self.pop()
            else:
                lo = mid

        heapq.heapify(self)

    def add(self: 'LimitedPQ', item: State):
        '''amortized O(log n), worst-case O(n)'''
        heapq.heappush(self, item)
        if len(self) >= 2 * self.max_size:
            self.kill_extra_elts()

    def best_score(self: 'LimitedPQ') -> float:
        '''O(1)'''
        if len(self) == 0:
            return float('inf')
        else:
            return self[0].score()

    def best(self: 'LimitedPQ') -> State:
        '''O(1)'''
        return self[0]

    def pop_best(self: 'LimitedPQ') -> State:
        '''O(log n)'''
        return heapq.heappop(self)

### Utils for the priority queue

T = TypeVar('T')

def partition(xs: list[T], lo: int, hi: int, pred: Callable[[T], bool]) -> int:
    '''O(hi - lo)'''
    assert 0 <= lo <= hi < len(xs)
    while lo < hi:
        if pred(xs[lo]):
            lo += 1
        else:
            xs[lo], xs[hi - 1] = xs[hi - 1], xs[lo]
            hi -= 1
    return lo

### The main search code

def search(examples: List[str], max_size: int):
    N = len(examples)
    pq = LimitedPQ(max_size)
    heuristic = BestHeuristic(examples)
    starting_index = (0,) * N
    ending_index = tuple(len(e) for e in examples)
    pq.add(State(
        indices=starting_index,
        score_so_far=0.,
        score=heuristic.value_at(starting_index),
        regex_so_far=[]
    ))
    # We have a loop
    while len(pq) > 0:
        state = pq.pop_best()
        if state.indices == ending_index:
            return state.regex_so_far
       #TODO: add the things after state

# Liza notes:
# 1) should we actually be ending if state,indices=ending_index what about 
# \d for 57 and 7 (are we just hoping it will not be outputed)
# 2) idea pop best state add all regexes s.t output a st


def next_regexes(examples: List[str], curr_states: Tuple(int)):
    # next characters
    next_chars = [example[index] for example, index in zip(examples, curr_states)]
    chars_left = [example[index:len(example)] for example, index in zip(examples, curr_states)]

    # posible constants
    longest_prefix = os.path.commonprefix(chars_left)
    constant_regxes = itertools.accumulate(map(lambda x: Constant(''.join([x,])), longest_prefix))

    # possible atomic regexes using Mark's code
    atomic_regexes_for_each = (set(atomic_regexes_matching(t)) for t in next_chars)
    regexes = functools.reduce(set.intersection, atomic_regexes_for_each)
    
    # possible optionals 
    if any(t == '' for t in texts):


if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)