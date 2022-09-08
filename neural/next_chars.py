from typing import *
import functools, itertools
import string
import numpy as np

# At the start it calls regex_to_nfa which returns the starting state for the NFA
# Then at each step it can ask for possible next chars, and whether it's at an
# ending state
# And to go to the next state it uses consume_a_char

class Node:
    is_end: bool
    transitions: Dict[str, 'Node']
    epsilon_transitions: Set['Node']

    def epsilon_closure(self) -> Set['Node']:
        result = set()
        self.epsilon_closure_union(result)
        return result

    def epsilon_closure_union(self, result: Set['Node']):
        '''Modify the set: result = result ∪ (the ε-closure of self)'''
        if self in result:
            return
        result.add(self)
        for other in self.epsilon_transitions:
            other.epsilon_closure_union(result)

    def __init__(self, is_end=False):
        self.is_end = is_end
        self.transitions = {}
        self.epsilon_transitions = set()

def possible_next_chars(state: Set[Node]) -> Set[str]:
    return {char for node in state for char in node.transitions.keys()}

def end_token_is_allowed_here(state: Set[Node]) -> bool:
    return any(n.is_end for n in state)

def consume_a_char(state: Set[Node], ch: str) -> Set[Node]:
    next_state = set()
    for node in state:
        if ch in node.transitions:
            node.transitions[ch].epsilon_closure_union(next_state)
    if len(next_state) == 0:
        raise Exception(f"you can't actually use that character ({ch})")
    return next_state

def regex_to_nfa(components: List[str]) -> Set[Node]:
    cur_node = Node(is_end=True)
    for component in reversed(components):
        cur_node = prepend_component_to_nfa(component, cur_node)
    return cur_node.epsilon_closure()

# helper function for regex_to_nfa
def prepend_component_to_nfa(component: str, next_node: Node) -> Node:
    classes = {
        '[0-9]': list(string.digits),
        '[a-z]': list(string.ascii_lowercase),
        '[A-Z]': list(string.ascii_uppercase),
        '[a-zA-Z]': list(string.ascii_letters),
        '[a-zA-Z0-9]': list(string.ascii_letters) + list(string.digits),
        '[0-9]+': list(string.digits),
        '[a-z]+': list(string.ascii_lowercase),
        '[A-Z]+': list(string.ascii_uppercase),
        '[a-zA-Z]+': list(string.ascii_letters),
        '[a-zA-Z0-9]+': list(string.ascii_letters) + list(string.digits),
    }
    if component in classes:
        chars = classes[component]
        if component[-1] == '+':
            # do plus stuff
            # node0 ----> node1 --ε--> next_node
            #              / ↑
            #              \_/
            # (ascii art by Mark Barbone July 19th, 2022 ™)
            node0 = Node()
            node1 = Node()
            node0.transitions = {char: node1 for char in chars}
            node1.transitions = {char: node1 for char in chars}
            node1.epsilon_transitions = {next_node}
            return node0
        else:
            # do not-plus stuff
            # node0 ----> next_node
            node0 = Node()
            node0.transitions = {char: next_node for char in chars}
            return node0
    else:
        # do literal character stuff
        # node0 ----> next_node
        assert len(component) == 1 # it's only literal characters for now
        node0 = Node()
        node0.transitions = {component: next_node}
        return node0


## ^^^ that codes makes an NFA
# Now we will make a DFA

class DFANode:
    node_id: int
    is_end: bool
    transitions: Dict[str, int]
    def __init__(self, node_id, is_end, transitions):
        self.node_id = node_id
        self.is_end = is_end
        self.transitions = transitions

class DFA:
    # map indices (node ids) to whether they're the end node + their transitions
    nodes: List[DFANode]

    def __init__(self, nfa: Set[Node]):
        # NFA to DFA: powerset construction
        self.nodes = [None]
        set_to_node: List[Tuple[Set[Node], int]] = [(nfa, 0)]
        worklist: List[Set[Node]] = [nfa]
        def set_to_id(s):
            for s1, i in set_to_node:
                if s == s1:
                    return i
            i = len(self.nodes)
            self.nodes.append(None)
            set_to_node.append((s, i))
            worklist.append(s)
            return i

        while len(worklist) > 0:
            s = worklist.pop()
            transitions: Dict[str, int] = dict()
            for char in possible_next_chars(s):
                s2 = consume_a_char(s, char)
                i2 = set_to_id(s2)
                transitions[char] = i2
            is_end = end_token_is_allowed_here(s)
            i = set_to_id(s)
            self.nodes[i] = DFANode(i, is_end, transitions)

    def sample(self):
        '''Samples uniformly from the DFA. Probably not what you want'''
        out = ''
        node = self.nodes[0]
        while True:
            ch = np.random.choice(np.array(list(node.transitions.keys()) +
                (['END'] if node.is_end else [])))
            if ch == 'END':
                return out
            out += ch
            node = self.nodes[node.transitions[ch][0]]

def error_location(regex: List[str], ex: str) -> int:
    state = regex_to_nfa(regex)
    for i, ch in enumerate(ex):
        if ch not in possible_next_chars(state):
            return i
        state = consume_a_char(state, ch)
    return -1



