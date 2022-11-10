from Heuristics import *
from typing import *
from Regex import *
import itertools, functools
import matplotlib.pyplot as plt

VSAState = Tuple[int, ...]

def atomic_regexes_matching_many(examples: List[str], start: VSAState, end: VSAState) -> Iterable[Regex]:
    texts = [e[a:b] for e, a, b in zip(examples, start, end)]
    nonempty_texts = [t for t in texts if t != '']
    atomic_regexes_for_each = (set(atomic_regexes_matching(t)) for t in nonempty_texts)
    regexes = functools.reduce(set.intersection, atomic_regexes_for_each)
    if any(t == '' for t in texts):
        return map(Optional, regexes)
    else:
        return regexes

def actual_scores(examples: List[str]) -> Dict[VSAState, float]:
    scores: Dict[VSAState, float] = {tuple(len(e) for e in examples): 0.}
    states = sorted(itertools.product(*(range(len(e)+1) for e in examples)), key=lambda s: -sum(s))
    for state in states:
        if state == tuple(len(e) for e in examples): continue
        later_states = [end
            for end in itertools.product(*(range(i,len(e)+1) for i, e in zip(state, examples)))
            if end != state
        ]
        def score(regex, end):
            return sum(regex.specificity_score(e[a:b]) for e, a, b in zip(examples, state, end)) \
                + regex.simplicity_score() + scores[end]
        scores[state] = min(score(regex, end) for end in later_states for regex in atomic_regexes_matching_many(examples, state, end))
    return scores

def heuristcs_scores(examples: List[str], type: str) -> Dict[VSAState, float]:
    if type == 'max':
        heuristic = MaxHeuristic(examples)
    elif type == 'sum':
        heuristic = SumHeuristic(examples)
    elif type == 'avg':
        heuristic = AverageHeuristic(examples)
    elif type == 'total':
        heuristic = TotalHeuristic(examples)
    else:
        raise Exception('type should be one of: max, sum, avg')
    return {
        state: heuristic.value_at(state)
        for state in itertools.product(*(range(len(e)+1) for e in examples))
    }



def graph_actual_vs_heuristics(first: Dict[VSAState, float], second: Dict[VSAState, float], types: List[str] ):
    order = first.keys()
    ordered_results = [second[item] for item in order]
    plt.scatter(first.values(), ordered_results)
    plt.xlabel(types[0] + ' scores')
    plt.ylabel(types[1] + ' scores')
    plt.show()
    

def graph_many_2D(results: list[Dict[VSAState, float]], types: list[str] ):
    order = results[0].keys()
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(results)):
        ordered_results = [results[i][item] for item in order]
        ax.scatter(results[0].values(), ordered_results, label = types[i])
    plt.xlabel('Actual scores')
    plt.ylabel('Hueristics scores')
    plt.legend(loc="upper left")
    plt.show()
    

def graph_things_3D_two_examples(results: Dict[VSAState, float]):
    state_one = [ item[0] for item in results.keys()]
    state_two = [ item[1] for item in results.keys()]
    scores = list(results.values())
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(state_one, state_two, scores)
    ax.set_xlabel('VSA State 1')
    ax.set_ylabel('VSA State 2')
    ax.set_zlabel('Scores')
    plt.show()
    

def graph_things_3D_two_examples_many_heuristics(results: list[Dict[VSAState, float]], types: List[str]):
    state_one = [ item[0] for item in results[0].keys()]
    state_two = [ item[1] for item in results[0].keys()]
    order = results[0].keys()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(results)):
        ordered_results = [results[i][item] for item in order]
        ax.scatter(state_one, state_two, ordered_results, label = types[i])
    ax.set_xlabel('VSA State 1')
    ax.set_ylabel('VSA State 2')
    ax.set_zlabel('Scores')
    plt.legend(loc="upper right")
    plt.show()

def graph_all_two_examples(examples: list[str]):
    max_scores = heuristcs_scores(examples, 'max')
    sum_scores = heuristcs_scores(examples, 'sum')
    avg_scores = heuristcs_scores(examples, 'avg')
    actual = actual_scores(examples)
    graph_things_3D_two_examples_many_heuristics([actual, max_scores,sum_scores, avg_scores], ['actual', 'max', 'sum', 'avg'])


def graph_vs(examples: list[str]):
    max_scores = heuristcs_scores(examples, 'max')
    sum_scores = heuristcs_scores(examples, 'sum')
    avg_scores = heuristcs_scores(examples, 'avg')
    total_scores = heuristcs_scores(examples, 'total')
    actual = actual_scores(examples)
    graph_many_2D([actual, max_scores,sum_scores, avg_scores, total_scores], ['actual', 'max', 'sum', 'avg', 'total'])