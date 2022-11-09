from typing import *
from VSA import VSAState
import math, itertools, functools, dataclasses

class PQItemProtocol(Protocol):
    vsa_state: VSAState
    score: float

Item = typing.TypeVar(bound=PQItemProtocol)

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

class LimitedPQ(Generic[State]):
    '''
    A mutable priority queue, that you add stuff to over time, with limited space
    '''

    heap: List[State]
    inds: Dict[VSAState, int]
    max_size: int

    def __init__(self: 'LimitedPQ', max_size: int):
        self.max_size = max_size
        self.heap = []
        self.inds = {}

    def __len__(self) -> int:
        return len(self.heap)

    def kill_extra_elts(self: 'LimitedPQ'):
        '''quickselect then re-heapify: O(n)'''
        if len(self) <= self.max_size: return

        self.inds = {}
        lo = 0
        heap = self.heap
        while len(heap) > self.max_size:
            cut_elt = heap[self.max_size]
            mid = partition(heap, lo, len(heap), lambda x: x.score < cut_elt.score)
            if mid >= self.max_size:
                while len(heap) > mid: heap.pop()
            else:
                lo = mid

        heapify(heap)
        self.inds = {x.vsa_state: i for i, x in enumerate(self.heap)}

    def add(self: 'LimitedPQ', item: State):
        '''amortized O(log n), worst-case O(n)'''
        if item.vsa_state in self.inds:
            ind = self.inds[item.vsa_state]
            if item.score >= self.heap[ind].score:
                return
            self.heap[ind] = item
            bubble_up_with_inds(self.heap, self.inds, ind)
            return
        i = len(self.heap)
        self.inds[item.vsa_state] = i
        self.heap.push(item)
        bubble_up_with_inds(self.heap, self.inds, i)
        if len(self.heap) >= 2 * self.max_size:
            self.kill_extra_elts()

    def best_score(self: 'LimitedPQ') -> float:
        '''O(1)'''
        if len(self.heap) == 0:
            return float('inf')
        else:
            return self.heap[0].score()

    def best(self: 'LimitedPQ') -> State:
        '''O(1)'''
        return self.heap[0]

    def pop_best(self: 'LimitedPQ') -> State:
        '''O(log n)'''
        if len(self.heap) == 1:
            self.inds = {}
            return self.heap.pop()
        item = self.heap[0]
        last = self.heap.pop()
        self.heap[0] = last
        del self.inds[item.vsa_state]
        self.inds[last.vsa_state] = 0
        bubble_down_with_inds(self.heap, self.inds, 0)
        return item

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

def bubble_down(xs: list[Item], i: int):
    '''O(log(n/i))'''
    while True:
        x = xs[i]
        l = 2*i + 1
        r = 2*i + 2
        if l < len(xs) and xs[l].score < x.score:
            xs[i] = xs[l]
            xs[l] = x
            i = l
        elif r < len(xs) and xs[r].score < x.score:
            xs[i] = xs[r]
            xs[r] = x
            i = r
        else:
            break

def heapify(xs: list[T]):
    '''O(n)'''
    for i in reversed(range((len(xs)+1) // 2)):
        bubble_down(xs, i)

def bubble_down_with_inds(xs: List[Item], inds: Dict[VSAState, int], i: int):
    '''O(log(n/i))'''
    while True:
        x = xs[i]
        l = 2*i + 1
        r = 2*i + 2
        if l < len(xs) and xs[l].score < x.score:
            inds[xs[l].vsa_state] = i
            inds[x.vsa_state] = l
            xs[i] = xs[l]
            xs[l] = x
            i = l
        elif r < len(xs) and xs[r].score < x.score:
            inds[xs[r].vsa_state] = i
            inds[x.vsa_state] = r
            xs[i] = xs[r]
            xs[r] = x
            i = r
        else:
            break

def bubble_up_with_inds(xs: list[Item], inds: Dict[VSAState, int], i: int):
    '''O(log i)'''
    while i > 0:
        x = xs[i]
        p = (i-1)//2
        if x.score < xs[p]:
            inds[xs[p].vsa_state] = i
            inds[x.vsa_state] = p
            xs[i] = xs[p]
            xs[p] = x
            i = p
        else:
            break

