from typing import *
import string

# At the start it calls regex to nfa which returns the starting state for the NFA
# Then at each step it can ask for possible next chars, and whether it's at an
# ending state
# And to go to the next state it uses consume_a_char

class Node:
    is_end: bool
    transitions: Dict[str, 'Node']
    epsilon_transitions: Set['Node']

    def epsilon_closure(self) -> Set['Node']:
        return self.epsilon_closure_helper(set())

    def epsilon_closure_helper(self, already_seen: Set['Node']) -> Set['Node']:
        if self in already_seen:
            return already_seen
        already_seen.add(self)
        for other in self.epsilon_transitions:
            already_seen = other.epsilon_closure_helper(already_seen)
        return already_seen

    def __init__(self):
        self.is_end = False
        self.transitions = {}
        self.epsilon_transitions = set()

def possible_next_chars(state: Set[Node]) -> Set[str]:
    result = set()
    for node in state:
        result |= set(node.transitions.keys())
    return result

def end_token_is_allowed_here(state: Set[Node]) -> bool:
    return any(n.is_end for n in state)

def consume_a_char(state: Set[Node], ch: str) -> Set[Node]:
    next_state = set()
    for node in state:
        if ch in node.transitions:
            next_state |= node.transitions[ch].epsilon_closure()
    if len(next_state) == 0:
        raise Exception(f"you can't actually use that character ({ch})")
    return next_state

def regex_to_nfa(components: List[str]) -> Set[Node]:
    end_node = Node()
    end_node.is_end = True
    cur_node = end_node
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
        #ssert len(component) == 1 # it's only literal characters for now
        node0 = Node()
        node0.transitions = {component: next_node}
        return node0
        