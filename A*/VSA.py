from typing import *
from Regex import *
import itertools, functools, dataclasses, math
from collections import defaultdict

### We need the VSA-based algorithm for computing the heuristics

VSAState = Tuple[int, ...]

def intersect_edge_sets(a: Set[Regex], b: Set[Regex]) -> Set[Regex]:
    # This is bad
    out = set()
    if len(b) < len(a): a, b = b, a
    for r in a:
        if r in b:
            out.add(r)
        elif isinstance(r, Optional):
            if r.contents in b:
                out.add(r)
        elif (opt := Optional(r)) in b:
            out.add(opt)
    return out

@dataclasses.dataclass
class VSA:
    examples: List[str]
    edges: Dict[VSAState, Dict[VSAState, Set[Regex]]] = \
            dataclasses.field(default_factory=lambda: defaultdict(dict))
    is_reversed: bool = False

    @staticmethod
    def single_example(example: str) -> 'VSA':
        res = VSA([example])
        for i in range(len(example)+1):
            for j in range(i+1, len(example)+1):
                res.update((i,), (j,), set(atomic_regexes_matching(example[i:j])))
        return res

    def reversed(self) -> 'VSA':
        assert not self.is_reversed
        res = VSA(self.examples)
        for s, edges in self.edges.items():
            for e, rs in edges.items():
                res.update(e, s, rs)
        res.is_reversed = True
        return res

    def update(self, s: VSAState, e: VSAState, edge: Set[Regex]):
        self.edges[e] # HACK (Mark)
        if e in self.edges[s]:
            self.edges[s][e] |= edge
        else:
            self.edges[s][e] = edge

    def merge(self, other: 'VSA') -> 'VSA':
        res = VSA(self.examples + other.examples)
        for s_a, a_edges in self.edges.items():
            for s_b, b_edges in other.edges.items():
                s = s_a + s_b
                # regular edges
                for e_a, r_a in a_edges.items():
                    for e_b, r_b in b_edges.items():
                        if len(both := intersect_edge_sets(r_a, r_b)) > 0:
                            res.update(s, e_a+e_b, both)
                # new optional edges
                for e_a, r_a in a_edges.items():
                    res.update(s, e_a+s_b, {r.opt() for r in r_a})
                for e_b, r_b in b_edges.items():
                    res.update(s, s_a+e_b, {r.opt() for r in r_b})

        return res

    def all_best_regexes(
            self,
            score_fn: Callable[[Regex, List[str]], float]
            ) -> Dict[VSAState, Tuple[float, Regex, VSAState]]:
        start_state = (0,) * len(self.examples)
        end_state = tuple(len(ex) for ex in self.examples)
        if self.is_reversed:
            start_state, end_state = end_state, start_state
        regexes = {end_state: (0., None, None)}

        # do DFS
        def get(start):
            if start in regexes: return regexes[start]
            cur_best_score = math.inf
            cur_best_r = None
            cur_best_end = None
            for end, rs in self.edges[start].items():
                # HACK (Mark): to support searching thru reversed VSAs as well
                texts = [ex[s:e] if s <= e else ex[e:s] for ex, s, e in zip(self.examples, start, end)]
                score, best_r = min((score_fn(r, texts), r) for r in rs)
                # if score == math.inf: print(texts, best_r, score_fn(best_r, texts))
                score += get(end)[0]
                if score < cur_best_score:
                    cur_best_score = score
                    cur_best_r = best_r
                    cur_best_end = end
            regexes[start] = cur_best_score, cur_best_r, cur_best_end
            return regexes[start]
        get(start_state)

        return regexes

    def single_best_regex(self, score_fn) -> List[Regex]:
        d = self.all_best_regexes(score_fn)
        regex = []
        state = (0,) * len(self.examples)
        end = tuple(len(ex) for ex in self.examples)
        while state != end:
            __, r, state = d[state]
            regex.append(r)
        return d[(0,) * len(self.examples)][0], regex

    def unsound_prune(self, heuristic: Callable[[VSAState], float], k: int) -> 'VSA':
        N = sum(len(e) for e in self.examples)
        buckets = {i: [] for i in range(N+1)}
        for s in self.edges:
            b = buckets[sum(s)]
            score = heuristic(s)
            # k is small, I hope
            if len(b) < k:
                b.append((s, score))
                if len(b) == k: b.sort(key=lambda x: x[1])
            elif b[-1][1] > score:
                i = next(i for i in reversed(range(k)) if i == 0 or b[i-1][1] <= score)
                b.insert(i, (s, score))
                b.pop()
        new_states = {s for b in buckets.values() for s, __ in b}
        res = VSA(self.examples)
        for s in new_states:
            for e, rs in self.edges[s].items():
                if e in new_states: res.update(s, e, rs)
        return res

    def scores(self, score_fn) -> Dict[VSAState, float]:
        fwd = self.all_best_regexes(score_fn)
        bwd = self.reversed().all_best_regexes(score_fn)
        return {s: fwd[s][0] + (bwd[s][0] if s in bwd else math.inf) for s in fwd}

    def sound_prune(self, score_fn, cutoff: float) -> 'VSA':
        cutoff += 0.001 # allow for rounding errors
        fwd = self.all_best_regexes(score_fn)
        bwd = self.reversed().all_best_regexes(score_fn)
        scores = {state: fwd[state][0] + (bwd[state][0] if state in bwd else math.inf) for state in fwd}
        res = VSA(self.examples)
        for s, edges in self.edges.items():
            if scores[s] > cutoff:
                continue
            for e, rs in edges.items():
                if scores[e] <= cutoff: res.update(s, e, rs)
        return res

def MDL(r: Regex, texts: List[str]) -> float:
    return r.simplicity_score() + sum(r.specificity_score(t) for t in texts)
import Heuristics
ex = ['mbarbone@ucsd.edu', 'epertsev@ucsd.edu', 'npolikarpova@eng.ucsd.edu', 'tberg@ucsd.edu']
vs = [VSA.single_example(e) for e in ex]
v = vs[0].merge(vs[1])

if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)
    vsas = [VSA.single_example(ex) for ex in inputs]
    big_vsa = functools.reduce(lambda a, b: a.merge(b), vsas)
    score, regex = big_vsa.single_best_regex(lambda r, ts: r.simplicity_score() +
                                 sum(r.specificity_score(t) for t in ts))
    print(f'{score}, {"".join(str(r) for r in regex)}')



