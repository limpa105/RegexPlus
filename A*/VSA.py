from typing import *
from Regex import *
import itertools, functools, dataclasses, math

### We need the VSA-based algorithm for computing the heuristics

VSAState = Tuple[int, ...]

@dataclasses.dataclass
class VSA:
    examples: List[str]
    edges: Dict[VSAState, Dict[VSAState, Set[Regex]]]

    @staticmethod
    def single_example(example: str) -> 'VSA':
        examples = [example]
        edges = {
                (i,): {(j,): set(atomic_regexes_matching(example[i:j]))
                            for j in range(i+1, len(example)+1)}
                for i in range(len(example)+1)
            }
        return VSA(examples=examples, edges=edges)

    def merge(self, other: 'VSA') -> 'VSA':
        examples = self.examples + other.examples
        # Basic edges
        edges = {}
        for s_a, a_edges in self.edges.items():
            for s_b, b_edges in other.edges.items():
                # regular edges
                outgoing_edges = {
                        e_a+e_b: both_regexes
                        for e_a, r_a in a_edges.items()
                        for e_b, r_b in b_edges.items()
                        if len(both_regexes := r_a & r_b) > 0
                    }
                # new optional edges
                for e_a, r_a in a_edges.items():
                    end = e_a + s_b
                    new_edges = {r.opt() for r in r_a}
                    if end not in outgoing_edges:
                        outgoing_edges[end] = new_edges
                    else:
                        outgoing_edges[end] |= new_edges
                for e_b, r_b in b_edges.items():
                    end = s_a + e_b
                    new_edges = {r.opt() for r in r_b}
                    if end not in outgoing_edges:
                        outgoing_edges[end] = new_edges
                    else:
                        outgoing_edges[end] |= new_edges
                edges[s_a+s_b] = outgoing_edges

        return VSA(examples, edges)

    def all_best_regexes(
            self,
            score_fn: Callable[[Regex, List[str]], float]
            ) -> Dict[VSAState, Tuple[float, Regex, VSAState]]:
        end_state = tuple(len(ex) for ex in self.examples)
        regexes = {end_state: (0., None, None)}

        # Need to process in a particular order -- could do DFS but I'm lazy
        for start, edges in sorted(self.edges.items(), key=lambda x: -sum(x[0])):
            if start == end_state: continue
            cur_best_score = math.inf
            cur_best_r = None
            cur_best_end = None
            for end, rs in edges.items():
                texts = [ex[s:e] for ex, s, e in zip(self.examples, start, end)]
                score, best_r = min((score_fn(r, texts), r) for r in rs)
                score += regexes[end][0]
                if score < cur_best_score:
                    cur_best_score = score
                    cur_best_r = best_r
                    cur_best_end = end
            regexes[start] = cur_best_score, cur_best_r, cur_best_end
        return regexes


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
    d = big_vsa.all_best_regexes(lambda r, ts: r.simplicity_score() +
                                 sum(r.specificity_score(t) for t in ts))
    print(f'result score is {d[(0,)*len(inputs)]}')
