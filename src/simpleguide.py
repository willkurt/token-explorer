from transformers import AutoTokenizer
import re
import time
from greenery import parse
import pickle
import hashlib
import os



class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_id = None
        
class TokenTrie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, token_meta):
        node = self.root
        word = token_meta['str']
        token_id = token_meta['id']
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id

    def collect_valid_tokens(self, state, fsm):
        node_state_stack = [(self.root, state)]
        valid_tokens = []
        while node_state_stack:
            node, s = node_state_stack.pop()
            for c, next_node in node.children.items():
                for cc, next_state in fsm.map[state].items():
                    if cc.accepts(c) and fsm.islive(next_state):
                        if next_node.token_id:
                            valid_tokens.append(next_node.token_id)
                        node_state_stack.append((next_node, next_state))
        return valid_tokens

class SimpleGuide:
    """
    The aim of this is to provide a low-memory guide without much regard to cost at inference time.
    Since we're iterating manually through token selection, it doesn't really matter if we're adding 
    a few 100ms to the inference time. I'm pretty sure this could be may much more efficient eventually.
    """
    def __init__(self, regex_struct, tokenizer, no_cache=False, verbose=False):
        self.regex_struct = re.compile(regex_struct)
        start_time = time.time()
        self.pattern = parse(regex_struct)
        end_time = time.time()
        if verbose:
            print(f"Regex parsing time: {(end_time - start_time) * 1000:.2f}ms")
        start_time = time.time()
        self.fsm = self.pattern.to_fsm()
        end_time = time.time()
        if verbose:
            print(f"FSM construction time: {(end_time - start_time) * 1000:.2f}ms")
        self.tokenizer = tokenizer
        # get the string representation of the tokenizer vocabulary
        self.vocab = tokenizer.get_vocab()
        self.vocab_list = [{
            'id': value,
            'str': tokenizer.decode(value)
            } for value in self.vocab.values()]
        self.string_so_far = ""
        self.tokens_so_far = []
        self.finished = False
        self.build_token_str_trie()
        self.build_state_token_map(no_cache, verbose)
    
    def build_token_str_trie(self):
        self.token_str_trie = TokenTrie()
        for item in self.vocab_list:
            self.token_str_trie.insert(item)

    def build_state_token_map(self, no_cache=False, verbose=False):
        # Create a unique hash for this regex and tokenizer combination
        cache_key = hashlib.md5(
            f"{self.regex_struct}_{self.tokenizer.name_or_path}".encode()
        ).hexdigest()
        
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        cache_file = os.path.join(cache_dir, f"state_token_map_{cache_key}.pkl")
        
        # Try to load from cache first
        if os.path.exists(cache_file) and not no_cache:
            try:
                with open(cache_file, 'rb') as f:
                    self.state_token_map = pickle.load(f)
                return
            except Exception as e:
                print(f"Cache load failed: {e}")
        
        # If cache doesn't exist or fails, build the map
        start_time = time.time()
        self.state_token_map = {}
        for state in self.fsm.states:
            self.state_token_map[state] = self.token_str_trie.collect_valid_tokens(state, self.fsm)
        end_time = time.time()
        if verbose:
            print(f"State token map construction time: {(end_time - start_time) * 1000:.2f}ms")
        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        if not no_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.state_token_map, f)
            except Exception as e:
                print(f"Cache save failed: {e}")

    def get_current_state(self, candidate, state=None):
        if state is None:
            state = self.fsm.initial
        for c in candidate:
            valid_next = [state for cc, state in self.fsm.map[state].items() if cc.accepts(c) and self.fsm.islive(state)]
            if len(valid_next) == 0:
                return None
            state = valid_next[0]
        return state
    
    def is_potential_prefix(self, candidate):
        state = self.get_current_state(candidate)
        return state is not None
    
    def get_tokens(self):
        """
        Appends the token to the string_so_far (temporarily) and returns the ids of the tokens that match the current regex
        given the string so far.
        """
        # Here's where we can distinguish between in a finished state and when dead.
        if self.finished:
            return [self.tokenizer.eos_token_id]
        matching_tokens = self.state_token_map[self.get_current_state(self.string_so_far)]
        return matching_tokens 

    def advance(self, token_id):
        if token_id == self.tokenizer.eos_token_id:
            self.finished = True
            return self
        self.string_so_far += self.tokenizer.decode(token_id)
        self.tokens_so_far.append(token_id)
        return self

    def is_finished(self):
        # Might want to also check if it's dead.
        finished=self.get_current_state(self.string_so_far) in self.fsm.finals
        dead=not self.fsm.islive(self.get_current_state(self.string_so_far))
        return finished or dead
    
    def is_dead(self):
        current_state = self.get_current_state(self.string_so_far)
        live_states = [val for val in self.fsm.map[current_state].values() if self.fsm.islive(val)]
        return len(live_states) == 0
        return not self.fsm.islive(current_state)

def test_guide_loading():
    import sys
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    print("Vocab size:", len(tokenizer.get_vocab()))
    start_time = time.time()
    #guide = SimpleGuide(r'(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/\d{4}', tokenizer)
    regex = r'\w{5} \w{5} \w{5}\n'
    #regex = r'Reasoning: [\w\s\d]{30,80}\n\nAnswer: 0?[1-9]\d{0,4}'
    guide = SimpleGuide(regex, tokenizer, no_cache=True, verbose=True)
    end_time = time.time()
    loading_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Guide loading time: {loading_time:.2f}ms")

    start_time = time.time()
    for _ in range(10000):
        guide.get_current_state("1")
    end_time = time.time()
    print(f"Current state time: {(end_time - start_time) * 1000 / 10000:.2f}ms")
    start_time = time.time()
    tokens = guide.get_tokens()
    end_time = time.time()
    #print([tokenizer.decode(token) for token in tokens])
    print(f"Tokens time: {(end_time - start_time) * 1000:.2f}ms")
    # get the size of the pickled guide
    print(f"SimpleGuide size: {sys.getsizeof(pickle.dumps(guide)) / 1024 / 1024:.2f}MB")

    print("Compare with Outlines-core")
    from outlines_core import Guide, Index, Vocabulary
    start_time = time.time()
    vocabulary = Vocabulary.from_pretrained(tokenizer.name_or_path)
    index = Index(regex, vocabulary)
    guide = Guide(index)
    end_time = time.time()
    print(f"Outlines-core time: {(end_time - start_time) * 1000:.2f}ms")
    # print the size of the guide in memory
    print(f"Outlines-core size: {sys.getsizeof(pickle.dumps(guide)) / 1024 / 1024:.2f}MB")

if __name__ == "__main__":
    test_guide_loading()
    #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    #guide = SimpleGuide("abc", tokenizer)
    #print(guide.get_current_state("abce"))