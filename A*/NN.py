import torch.nn.functional as F
from listener import *
from model import *
from Regex import *
import typing, dataclasses, functools, heapq

PLACE_HOLDER = ['[a-z]']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
encoder = Encoder(text_lang.n_words, hidden_size).to(device)
decoder = Decoder(hidden_size, regex_lang.n_words, Bilinear(hidden_size)).to(device)
file_name = 'opts-v3 2022-10-10 11:34:50.706173.pt'
encoder.load_state_dict(torch.load('encoder-' + file_name, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('decoder-' + file_name, map_location=torch.device('cpu')))
encoder = encoder.to(device=device)
decoder = decoder.to(device=device)
decoder.eval()
encoder.eval()

SECONDARY_SOLNS_COUNT = 10


@dataclasses.dataclass(eq=False)
class NNState:
    hidden: torch.Tensor
    cur_token: torch.Tensor
    example_length: torch.Tensor
    encoder_history: torch.Tensor

    @classmethod
    def tok(self, index: int):
        return torch.tensor([index], device=device).expand(1)

    @classmethod
    def initial(self, examples: List[str]):
        example_tensor, regex_tensor, example_length, regex_lengths = NNState.to_tensor(examples)
        encoder_history, hidden = encoder(example_tensor, example_length)
        return NNState(
            hidden = hidden,
            cur_token = NNState.tok(SOS_token),
            example_length = example_length,
            encoder_history = encoder_history,
        )

    def best_next_edges(self, k: int) -> List[Tuple[Regex, "NNState"]]:
        decoder_output, hidden, __ = decoder(
                    self.cur_token,
                    self.hidden,
                    self.encoder_history,
                    self.example_length)
        decoder_output = F.log_softmax(decoder_output, dim=1)
        top_probs, top_indexes = torch.topk(decoder_output[0], k)
        top_results = []
        for index in top_indexes:
            index = index.item()
            char = regex_lang.index2word[index]
            if char == '(+)':
                result = self.unplusify(hidden, char, optional=False)
                if result is not None:
                    top_results.append(result)
            elif char == '(*)':
                result = self.unplusify(hidden, char, optional=True)
                if result is not None:
                    top_results.append(result)
            elif char == 'OPT(':
                for result in self.unoptionify(hidden, char):
                    top_results.append(result)
            elif char in (')OPT', 'EOS'):
                continue
            else:
                new_state = NNState(
                    hidden = hidden,
                    cur_token = NNState.tok(index),
                    encoder_history = self.encoder_history,
                    example_length = self.example_length
                )
                if char in CHAR_CLASS_DICT:
                    top_results.append((CHAR_CLASS_DICT[char], new_state))
                else:
                    assert len(char) == 1, f'{char} should be a literal character'
                    top_results.append((Constant(char), new_state))
        return top_results


    def unplusify(self, hidden, prev_token: str, optional: bool) -> typing.Optional[Tuple[Regex, "NNState"]]:
        decoder_output, hidden, __ = decoder(
                NNState.tok(regex_lang.word2index[prev_token]),
                hidden,
                self.encoder_history,
                self.example_length)
        decoder_output = F.log_softmax(decoder_output, dim=1)
        __, index = torch.topk(decoder_output[0],1)
        index = index.item()
        token = regex_lang.index2word[index]
        if token not in REPEATED_CHAR_CLASS_DICT:
            return None
        new_state = NNState(
            hidden = hidden,
            cur_token = NNState.tok(index),
            encoder_history = self.encoder_history,
            example_length = self.example_length
        )
        if optional:
            return Optional(REPEATED_CHAR_CLASS_DICT[token]), new_state
        else:
            return REPEATED_CHAR_CLASS_DICT[token], new_state

    def unoptionify(self, hidden, token) -> Iterable[Tuple[Regex, "NNState"]]:
        assert token == 'OPT('
        pq = [(0., [], token, hidden)]
        count = 0
        while len(pq) > 0 and count < SECONDARY_SOLNS_COUNT:
            score, answer_so_far, token, hidden = heapq.heappop(pq)
            decoder_output, hidden, __ = decoder(
                NNState.tok(regex_lang.word2index[token]),
                hidden,
                self.encoder_history,
                self.example_length)
            decoder_output = F.log_softmax(decoder_output, dim=1)
            __, indices = torch.topk(decoder_output[0], SECONDARY_SOLNS_COUNT)
            for index in indices:
                token = regex_lang.index2word[index.item()]
                if token in ['OPT(', '(+)', '(*)', 'EOS']:
                    continue
                if token in CHAR_CLASS_DICT:
                    # Only want one character class in an optional
                    if len(answer_so_far) != 0: continue
                    # give it the ')OPT'
                    __, hidden, __ = decoder(
                            NNState.tok(index),
                            hidden,
                            self.encoder_history,
                            self.example_length)
                    yield (
                        Optional(CHAR_CLASS_DICT[token]),
                        NNState(
                            hidden = hidden,
                            cur_token = NNState.tok(regex_lang.word2index[')OPT']),
                            example_length = self.example_length,
                            encoder_history = self.encoder_history)
                        )
                    count += 1
                elif token == ')OPT':
                    yield (
                        Optional(Constant(''.join(answer_so_far))),
                        NNState(
                            hidden = hidden,
                            cur_token = NNState.tok(index),
                            example_length = self.example_length,
                            encoder_history = self.encoder_history)
                        )
                    count += 1
                else:
                    assert len(token) == 1
                    new_score = score - decoder_output[0, index.item()]
                    new_answer_so_far = answer_so_far + [token]
                    heapq.heappush(pq, (new_score, new_answer_so_far, token, hidden))

    @classmethod
    def to_tensor(self, examples:List[str]) -> torch.tensor:
        return pairs_to_tensors([(PLACE_HOLDER, examples)])

