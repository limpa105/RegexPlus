from typing import *
from VSA import VSAState
import math, itertools, functools, dataclasses

class PQItemProtocol(Protocol):
    indices: VSAState
    score: float

Item = TypeVar('Item', bound=PQItemProtocol)

class PQ(Protocol[Item]):
    '''
    The protocol that is implemented by:
     - PriorityQueue
     - Amortized Limited memory priority queue
     - MinMax heap limited memory priority queue
    '''
    def add(self, elt: Item):
        ...
    def pop_best(self) -> Item:
        ...

class LimitedPQ(Generic[Item]):
    '''
    A mutable priority queue, that you add stuff to over time, with limited space
    '''

    heap: List[Item]
    inds: Dict[VSAState, int]
    max_size: int

    def __init__(self: 'LimitedPQ', max_size: int):
        self.max_size = max_size
        self.heap = []
        self.inds = {}

    def __len__(self) -> int:
        return len(self.heap)

    def assert_invariants(self):
        # Sizes match up
        assert len(self.heap) == len(self.inds) <= self.max_size
        for i, x in enumerate(self.heap):
            # Indices are tracked
            assert self.inds[x.indices] == i
            # Heap invariant
            if i == 0: continue
            p = (i-1)//2
            assert self.heap[p].score <= x.score, f'{i=}, {p=}'

    def kill_extra_elts(self: 'LimitedPQ'):
        '''quickselect then re-heapify: O(n)'''
        if len(self) <= self.max_size: return

        self.inds = {}
        lo = 0
        heap = self.heap
        while len(heap) > self.max_size:
            cut_elt = heap[self.max_size]
            heap[self.max_size] = heap[lo]
            mid = partition(heap, lo + 1, len(heap), lambda x: x.score < cut_elt.score)
            heap[lo] = heap[mid - 1]
            heap[mid - 1] = cut_elt
            if mid >= self.max_size:
                while len(heap) > mid - 1: heap.pop()
            else:
                lo = mid

        heapify(heap)
        self.inds = {x.indices: i for i, x in enumerate(self.heap)}

    def add(self: 'LimitedPQ', item: Item):
        '''amortized O(log n), worst-case O(n)'''
        if item.indices in self.inds:
            ind = self.inds[item.indices]
            if item.score >= self.heap[ind].score:
                return
            self.heap[ind] = item
            bubble_up_with_inds(self.heap, self.inds, ind)
            return
        i = len(self.heap)
        self.inds[item.indices] = i
        self.heap.append(item)
        bubble_up_with_inds(self.heap, self.inds, i)
        if len(self.heap) >= 2 * self.max_size:
            self.kill_extra_elts()

    def best_score(self: 'LimitedPQ') -> float:
        '''O(1)'''
        if len(self.heap) == 0:
            return float('inf')
        else:
            return self.heap[0].score()

    def best(self: 'LimitedPQ') -> Item:
        '''O(1)'''
        return self.heap[0]

    def pop_best(self: 'LimitedPQ') -> Item:
        '''O(log n)'''
        if len(self.heap) == 1:
            self.inds = {}
            return self.heap.pop()
        item = self.heap[0]
        last = self.heap.pop()
        self.heap[0] = last
        del self.inds[item.indices]
        self.inds[last.indices] = 0
        bubble_down_with_inds(self.heap, self.inds, 0)
        return item

### Utils for the priority queue

T = TypeVar('T')

def partition(xs: List[T], lo: int, hi: int, pred: Callable[[T], bool]) -> int:
    '''O(hi - lo)'''
    assert 0 <= lo <= hi <= len(xs)
    while lo < hi:
        if pred(xs[lo]):
            lo += 1
        else:
            xs[lo], xs[hi - 1] = xs[hi - 1], xs[lo]
            hi -= 1
    return lo

def bubble_down(xs: List[Item], i: int):
    '''O(log(n/i))'''
    while True:
        x = xs[i]
        l = 2*i + 1
        r = 2*i + 2
        if l >= len(xs): break
        child = r if r < len(xs) and xs[r].score < xs[l].score else l
        if xs[child].score < x.score:
            xs[i] = xs[child]
            xs[child] = x
            i = child
        else:
            break

def heapify(xs: List[Item]):
    '''O(n)'''
    for i in reversed(range((len(xs)+1) // 2)):
        bubble_down(xs, i)

def bubble_down_with_inds(xs: List[Item], inds: Dict[VSAState, int], i: int):
    '''O(log(n/i))'''
    while True:
        x = xs[i]
        l = 2*i + 1
        r = 2*i + 2
        if l >= len(xs): break
        child = r if r < len(xs) and xs[r].score < xs[l].score else l
        if xs[child].score < x.score:
            inds[xs[child].indices] = i
            inds[x.indices] = child
            xs[i] = xs[child]
            xs[child] = x
            i = child
        else:
            break

def bubble_up_with_inds(xs: List[Item], inds: Dict[VSAState, int], i: int):
    '''O(log i)'''
    while i > 0:
        x = xs[i]
        p = (i-1)//2
        if x.score < xs[p].score:
            inds[xs[p].indices] = i
            inds[x.indices] = p
            xs[i] = xs[p]
            xs[p] = x
            i = p
        else:
            break

