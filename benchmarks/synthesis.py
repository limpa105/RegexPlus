import math
import re
import sys
from typing import List, Dict, Tuple, Set
import itertools
import pickle
import numpy as np 
import warnings
from regexobj import *
# ignoring sklearn warnings 
warnings.filterwarnings('ignore')

class VSA:
    num_nodes: int
    edges: List[Dict[int, Tuple[bool, Set[str]]]]
    start_node: int
    end_node: int

    def __init__(self, num_nodes, edges, start_node, end_node):
        self.num_nodes = num_nodes
        self.edges = edges
        self.start_node = start_node
        self.end_node = end_node

    def debug(self):
        print(f"VSA with {self.num_nodes} nodes")
        print(f"start: {self.start_node}, end: {self.end_node}")
        for i, e in enumerate(self.edges):
            print(f"  {i}: {e}")

def add_backslashes(s):
    return ''.join([disambiguate_char(c) for c in s])

def disambiguate_char(c):
    return '\\'*(c in '[]()\{\}*+?|^$.\\') + c

regparts_base = [r'[0-9]', r'[a-z]', r'[A-Z]', r'[a-zA-Z]', r'[a-zA-Z0-9]', r'\s', r'(\w+ ?)']
reg_strs = []
for part in regparts_base:
    reg_strs.append(part)
    reg_strs.append(part+'+')

def possible_regex_tokens(s):
    # FIXME: there should be a better way to represent regex tokens than strings
    yield add_backslashes(s) # constant string always works
    for reg_str in reg_strs:
        if (re.compile(reg_str+r'$')).match(s):
            yield reg_str
    # TODO: other tokens (capitalized words, stuff like that)

def mk_vsa(s: str) -> VSA:
    num_nodes = len(s) + 1
    edges = []
    for i in range(len(s)):
        outgoing_edges = {}
        for j in range(i+1, len(s)+1):
            outgoing_edges[j] = False, set(possible_regex_tokens(s[i:j]))
        edges.append(outgoing_edges)
    edges.append({})
    start_node = 0
    end_node = len(s)
    return VSA(num_nodes, edges, start_node, end_node)


# Intersect two VSAs, only retaining reachable nodes, and deduplicating the VSA
# with congruence closure
def intersect(va: VSA, vb: VSA) -> VSA:
    edges = []
    def new_node_id(es):
        node_id = len(edges)
        edges.append(es)
        return node_id
    memo = {}
    children_to_node_map = {}
    def dfs(a, b):
        if (a, b) in memo:
            return memo[a, b]
        if a == va.end_node and b == vb.end_node:
            node_id = new_node_id({})
            memo[a, b] = True, node_id
            return True, node_id

        children = []
        def process_edge(a_target, b_target, edge):
            can_reach_end, target = dfs(a_target, b_target)
            if can_reach_end:
                children.append((target, edge))

        # regular edges
        for a_target, a_edge in va.edges[a].items():
            for b_target, b_edge in vb.edges[b].items():
                is_opt = a_edge[0] or b_edge[0]
                e = a_edge[1].intersection(b_edge[1])
                if len(e) > 0:
                    process_edge(a_target, b_target, (is_opt, e))
        if USE_OPTIONALS:
            # optional edges where a is constant
            for b_target, b_edge in vb.edges[b].items():
                process_edge(a, b_target, (True, b_edge[1]))
            # optional edges where b is constant
            for a_target, a_edge in va.edges[a].items():
                process_edge(a_target, b, (True, a_edge[1]))

        if len(children) == 0:
            memo[a, b] = False, -1
            return False, -1
        children.sort()
        ns = tuple(n for n, e in children)
        es = tuple(e for n, e in children)
        if ns not in children_to_node_map:
            children_to_node_map[ns] = []
        for candidate in children_to_node_map[ns]:
            if candidate[0] == es:
                new_node = candidate[1]
                memo[a, b] = True, new_node
                return True, new_node
        node_id = new_node_id(dict(children))
        children_to_node_map[ns].append((es, node_id))
        memo[a, b] = True, node_id
        return True, node_id

    can_reach_end, start_node = dfs(va.start_node, vb.start_node)
    if not can_reach_end:
        raise Exception("No possible regex :(")
    num_nodes = len(edges)
    end_node = memo[va.end_node, vb.end_node][1]

    return VSA(num_nodes, edges, start_node, end_node)


def possible_regexes(v: VSA):
    def regexes_starting_at(a):
        if a == v.end_node:
            yield ""
            return
        for b, regexes in v.edges[a].items():
            for r in regexes:
                for rest in regexes_starting_at(b):
                    yield r + rest
    yield from regexes_starting_at(v.start_node)

token_weights: Dict[str, float] = {
    '\\s': -4,
    '\\s+': 1,  # 3
    '[0-9]': -4,
    '[0-9]+': 1,  # 10
    '[a-z]': -4,
    '[a-z]+': 1,  # 26
    '[A-Z]': -4,
    '[A-Z]+': 1,  # 26
    '[a-zA-Z]': -3,
    '[a-zA-Z]+': 2,  # 26 + 26 + 1
    '[a-zA-Z0-9]': -2,
    '[a-zA-Z0-9]+': 3,  # 26 + 26 + 10 + 1
    '(\\w ?)': -1,
    '(\\w ?)+': 4  # 100
}

# this determines parse order... yeck
ordered_tokens = [
    '\\s',
    '\\s+',
    '[0-9]',
    '[0-9]+',
    '[a-z]',
    '[a-z]+',
    '[A-Z]',
    '[A-Z]+',
    '[a-zA-Z]',
    '[a-zA-Z]+',
    '[a-zA-Z0-9]',
    '[a-zA-Z0-9]+',
    '(\\w ?)',
    '(\\w ?)+'
]

def wt_of_token(tok: Set[str], const_prob: float) -> Tuple[float, str]:
    special_things = set(reg_strs)
    if len(tok.difference(special_things)) > 0:
        # it has a literal string
        s, = tok.difference(special_things)
        return -26*const_prob, s
    else:
        for t in ordered_tokens:
            if t in tok:
                return token_weights[t], t            
    return math.inf, ""
    # # set is probably empty
    # assert len(tok) == 0
    # # there is no regex
    # raise Exception('no possible regex')

def regex_normalize(reg: RegexObj) -> RegexObj:
    """
    If anything matched by R is matched by S, 
    then R?S+ is equivalent to S+.
    """
    if reg.kind == "concat":
        if (reg.left.kind == "optional") and \
           (reg.right.kind == "plus"):
            pass
    
    # TODO: greenery? brzozowski derivs?

def get_best_regexes(v: VSA, const_prob:float, opt_prob:float, k=5) -> List[Tuple[float, str]]:
    '''Return the top k regexes. By default k = 5'''
    best_from_node = { v.end_node: [(0, "")] }
    def dfs(a):
        if a in best_from_node:
            return best_from_node[a]
        else:
            # make a list of all possibilities
            cur_best = set()
            for b, regexes in v.edges[a].items():
                wt, regex = wt_of_token(regexes[1], const_prob)
                if regexes[0]:
                    # extra weight for ?
                    wt += 52*opt_prob + len(regex) 
                    for wt_of_b, regex_of_b in dfs(b):
                        cur_best.add((wt + wt_of_b, '(' + regex + ')?' + regex_of_b))
                else:
                    for wt_of_b, regex_of_b in dfs(b):
                        cur_best.add((wt + wt_of_b, regex + regex_of_b))
            # get only the top k
            best = sorted(cur_best)[:k]
            best_from_node[a] = best
            return best
    return dfs(v.start_node)


def synthesize(inputs):
    
    # getting features from the examples 
    # getting the variance
    std = np.asarray([ len(j) for j in inputs]).std() 
    # getting the mean
    mean_len = np.asarray([ len(j) for j in inputs]).mean()
    dif = np.asarray([len(j) for j in inputs]).max() - np.asarray([len(j) 
            for j in inputs]).min()
    # length of shared constants 
    ans = inputs[0]
    for j in inputs:
        ans = set(j).intersection(ans)
    shared_count = len(ans)

    from numpy import linalg as LA
    def cosine_sim(v1,v2):
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        index = np.argmax([len(v1), len(v2)])
        if index==0:
            v2 = np.pad(v2,(0,len(v1)-len(v2)),'constant')
        else:
            v1 = np.pad(v1,(0,len(v2)-len(v1)),'constant')
        return np.dot(v1,v2)/(LA.norm(v1)*LA.norm(v2))

    test = [list(j) for j in inputs]
    test =  [[ord(i) for i in j] for j in test]
    test = [cosine_sim(i, test[0]) for i in test]
    cosine_sim = np.mean(test[1:])

    # loading the two models from files
    constants_model = pickle.load(open("constants_model.sav", 'rb'))
    optionals_model = pickle.load(open("optionals_model.sav", 'rb'))
    if cosine_sim == np.inf or cosine_sim == -np.inf or math.isnan(cosine_sim):
        cosine_sim = 0

    # predicting the probability that the regex will include a constant 
    const_prob = constants_model.predict_proba([[std,shared_count,dif, cosine_sim]])[0][1]

    #predicting the probability that the regex will NOT include an optional
    opt_prob = optionals_model.predict_proba([[shared_count,std,cosine_sim]])[0][0]

    # print("Probability of a constant", round(const_prob, 3))
    # print("Probability of NOT an optional", round(opt_prob, 3))



    vsa = mk_vsa(inputs[0])
    # print("there are %d nodes" % vsa.num_nodes)
    for s in inputs[1:]:
        vsa = intersect(vsa, mk_vsa(s))
        # print("there are %d nodes" % vsa.num_nodes)

    regexes = get_best_regexes(vsa, const_prob, opt_prob)
    # print(f"Best regex: {regex} (weight {wt})")
    return regexes

USE_OPTIONALS = True  # lol
if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)

    # Only enable regexes with `?` if the -q flag is used
    # global USE_OPTIONALS
    if len(sys.argv) > 1:
        if sys.argv[1] == '-q':
            USE_OPTIONALS = True

    print("doin' VSA stuff")

    for i, (wt, regex) in enumerate(synthesize(inputs)):
        print(f"  {i+1}. {regex}")
