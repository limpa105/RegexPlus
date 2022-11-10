import torch.nn.functional as F
from listener import *
from model import * 
from Regex import *
import typing, dataclasses

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

@dataclasses.dataclass
class NNState: 
    hidden: torch.Tensor
    cur_token: torch.Tensor
    example_length: torch.Tensor
    encoder_history: torch.Tensor


    @classmethod
    def initial(self, examples: List[str]):
        example_tensor, regex_tensor, example_length, regex_lengths = NNState.to_tensor(examples)
        encoder_history, hidden = encoder(example_tensor, example_length)
        cur_token = torch.tensor([SOS_token], device=device).expand(1)
        return NNState(
            hidden = hidden,
            cur_token = cur_token,
            example_length = example_length,
            encoder_history = encoder_history,
        )

    def best_next_edges(self, k: int) -> List[Tuple[Regex, "NNState"]]:
        decoder_output, self.hidden, attn_weights = decoder(
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
                result = self.unplusify(char, optional=False)
                if result is not None:
                    top_results.append(result)
            elif char == '(*)':
                result = self.unplusify(char, optional=True)
                if result is not None:
                    top_results.append(result)
            elif char == 'OPT(':
                result = self.unoptionofy(char)
                if result is not None:
                    top_results.append(result)
            elif char == ')OPT':
                continue 
            else:
                new_state = NNState(
                    hidden = self.hidden,
                    cur_token = torch.tensor([index], device=device).expand(1),
                    encoder_history = self.encoder_history,
                    example_length = self.example_length
                )
                top_results.append((Constant(char), new_state))
        return top_results
    

    def unplusify(self, prev_token: str, optional: bool) -> typing.Optional[Tuple[Regex, "NNState"]]:
        decoder_output, hidden, attn_weights = decoder(
                torch.tensor([regex_lang.word2index[prev_token]], device=device).expand(1),
                self.hidden,
                self.encoder_history,
                self.example_length)
        decoder_output = F.log_softmax(decoder_output, dim=1)
        _, index = torch.topk(decoder_output[0],1)
        index = index.item()
        token = regex_lang.index2word[index]
        if token not in REPEATED_CHAR_CLASS_DICT:
            return None
        new_state = NNState(
            hidden = hidden,
            cur_token = torch.tensor([index], device=device).expand(1),
            encoder_history = self.encoder_history,
            example_length = self.example_length
        )
        if optional:
            return Optional(REPEATED_CHAR_CLASS_DICT[token]), new_state
        else:
            return REPEATED_CHAR_CLASS_DICT[token], new_state            

    def unoptionofy(self, token) -> typing.Optional[Tuple[Regex, torch.Tensor]]:
        answer = []
        count = 0
        while token != 'OPT)' and count < 15:
            decoder_output, hidden, attn_weights = decoder(
                torch.tensor([regex_lang.word2index[token]], device=device).expand(1),
                self.hidden,
                self.encoder_history,
                self.example_length)
            decoder_output = F.log_softmax(decoder_output, dim=1)
            # potential make this beam later
            _, index = torch.topk(decoder_output[0],1)
            index = index.item()
            token = regex_lang.index2word[index]
            if token == '(OPT' or '+':
                return None 
            # can only have one charcacter class 
            try:
                answer.append(CHAR_CLASS_DICT[token])
                if len(answer)!=0:
                    return None 
                else:
                    return(Optional(answer[0]), hidden)
            except:
                answer.append(token)
        if count > 15:
            return None
        else: 
            return (Optional(Constant(''.join(answer))), hidden)


    @classmethod
    def to_tensor(self, examples:List[str]) -> torch.tensor:
        return pairs_to_tensors([(PLACE_HOLDER, examples)])

