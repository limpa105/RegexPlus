import math
import re
import sys
from typing import List, Dict, Tuple, Set
import itertools
import pickle
import numpy as np 
import warnings
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

def possible_regex_tokens(s):
    # FIXME: there should be a better way to represent regex tokens than strings
    yield add_backslashes(s) # constant string always works
    decis = re.compile(r'[0-9]+$')
    lows = re.compile(r'[a-z]+$')
    ups = re.compile(r'[A-Z]+$')
    alphas = re.compile(r'[a-zA-Z]+$')
    alnums = re.compile(r'[a-zA-Z0-9]+$')
    whites = re.compile(r'\s+$')
    words = re.compile(r'(\w+ ?)+$')
    if decis.match(s):
        yield "[0-9]+"
    if lows.match(s):
        yield "[a-z]+"
    if ups.match(s):
        yield "[A-Z]+"
    if alphas.match(s):
        yield "[a-zA-Z]+"
    if alnums.match(s):
        yield "[a-zA-Z0-9]+"
    if whites.match(s):
        yield "\\s+"
    if words.match(s):
        yield "(\w+ ?)+"
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


def wt_of_token(tok: Set[str], const_prob: float) -> Tuple[float, str]:
    special_things = {"[0-9]+", "[a-z]+", "[A-Z]+", "[a-zA-Z]+", "[a-zA-Z0-9]+", "\\s+", "(\w+ ?)+"}
    if len(tok.difference(special_things)) > 0:
        # it has a literal string
        s, = tok.difference(special_things)
        return -26*const_prob, s
    elif "\\s+" in tok:
        return 3, "\\s+"
    elif "[0-9]+" in tok:
        return 10, "[0-9]+"
    elif "[a-z]+" in tok:
        return 26, "[a-z]+"
    elif "[A-Z]+" in tok:
        return 26, "[A-Z]+"
    elif "[a-zA-Z]+" in tok:
        return 26 + 26, "[a-zA-Z]+"
    elif "[a-zA-Z0-9]+" in tok:
        return 26 + 26 + 10, "[a-zA-Z0-9]+"
    elif "(\w+ ?)+" in tok:
        return 100, "(\w+ ?)+"  # TODO: cardinality doesn't handle weight!
    else:
        return math.inf, ""
        # # set is probably empty
        # assert len(tok) == 0
        # # there is no regex
        # raise Exception('no possible regex')


def get_best_regex(v: VSA, const_prob:float, opt_prob:float) -> str:
    best_from_node = { v.end_node: (0, "") }
    def dfs(a):
        if a in best_from_node:
            return best_from_node[a]
        else:
            cur_best_wt = math.inf
            cur_best_regex = ""
            for b, regexes in v.edges[a].items():
                wt_of_b, regex_of_b = dfs(b)
                wt, regex = wt_of_token(regexes[1], const_prob)
                if regexes[0]: # extra weight for ? -- 50 is ok
                    wt += 52*opt_prob + len(regex) 
                if wt + wt_of_b < cur_best_wt:
                    cur_best_wt = wt + wt_of_b
                    if regexes[0]:
                        cur_best_regex = regex + '?' + regex_of_b
                    else:
                        cur_best_regex = regex + regex_of_b
            best_from_node[a] = cur_best_wt, cur_best_regex
            return cur_best_wt, cur_best_regex
    return dfs(v.start_node)


def main():
    # Only enable regexes with `?` if the -q flag is used
    global USE_OPTIONALS
    if len(sys.argv) > 1:
        if sys.argv[1] == '-q':
            USE_OPTIONALS = True

    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)

    print("doin' VSA stuff")
    
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

    # predicting the probability that the regex will include a constant 
    const_prob = constants_model.predict_proba([[std,shared_count,dif, cosine_sim]])[0][1]

    #predicting the probability that the regex will NOT include an optional
    opt_prob = optionals_model.predict_proba([[shared_count,std,cosine_sim]])[0][0]

    print("Probability of a constant", round(const_prob, 3))
    print("Probability of NOT an optional", round(opt_prob, 3))



    vsa = mk_vsa(inputs[0])
    # print("there are %d nodes" % vsa.num_nodes)
    for s in inputs[1:]:
        vsa = intersect(vsa, mk_vsa(s))
        # print("there are %d nodes" % vsa.num_nodes)

    wt, regex = get_best_regex(vsa, const_prob, opt_prob)
    print(f"Best regex: {regex} (weight {wt})")

USE_OPTIONALS = False
if __name__ == '__main__':
    main()
