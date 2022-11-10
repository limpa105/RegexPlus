from typing import *
import dataclasses

@dataclasses.dataclass(eq=False, repr=False)
class Node:
    is_end: bool = False
    transitions: Dict[str, 'Node'] = dataclasses.field(default_factory=dict)
    epsilon_transitions: Set['Node'] = dataclasses.field(default_factory=set)

    def epsilon_closure(self) -> FrozenSet['Node']:
        result = set()
        self.epsilon_closure_union(result)
        return frozenset(result)

    def epsilon_closure_union(self, result: Set['Node']):
        '''Modify the set: result = result ∪ (the ε-closure of self)'''
        if self in result:
            return
        result.add(self)
        for other in self.epsilon_transitions:
            other.epsilon_closure_union(result)

State = FrozenSet[Node]

def possible_next_chars(state: State) -> Set[str]:
    return {char for node in state for char in node.transitions.keys()}

def end_token_is_allowed_here(state: State) -> bool:
    return any(n.is_end for n in state)

def maybe_consume_a_char(state: State, ch: str) -> Optional[State]:
    next_state = set()
    for node in state:
        if ch in node.transitions:
            node.transitions[ch].epsilon_closure_union(next_state)
    if len(next_state) == 0:
        return None
    return frozenset(next_state)

def consume_a_char(state: State, ch: str) -> State:
    result = maybe_consume_a_char(state, ch)
    if result is None:
        raise Exception(f"you can't actually use that character ({ch})")
    return result

def matching_locations(state: State, text: str, starting_from: int) -> Iterator[int]:
    for i in range(starting_from, len(text)):
        if end_token_is_allowed_here(state):
            yield i
        char = text[i]
        state = maybe_consume_a_char(state, char)
        if state is None:
            return

def matches(state: State, text: str) -> bool:
    for char in text:
        state = maybe_consume_a_char(state, char)
        if state is None:
            return False
    return end_token_is_allowed_here(state)
