"""
This code is used to process an LLM one token at at time.

The Explorer class manages the prompt internally and handles all interactions with the LLM.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from src.simpleguide import SimpleGuide
class Explorer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        """
        Initialize the Explorer with a model name.
        
        Args:
            model_name: Name of the model to load (default "Qwen/Qwen2.5-0.5B")
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Auto select device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.guide = None
        
        # Initialize with empty promp
        self.prompt_text = ""
        self.prompt_tokens = []
    

    def clear_guide(self):
        self.guide = None

    def set_guide(self, regex_struct,ff_from=None):
        self.clear_guide()
        self.guide = SimpleGuide(regex_struct, self.tokenizer)
        if ff_from is not None:
            for token in self.prompt_tokens[ff_from:]:
                self.guide.advance(token)


    def set_prompt(self, prompt_text):
        """
        Set the current prompt text and update the encoded tokens.
        
        Args:
            prompt_text: The prompt text to set
        """
        self.prompt_text = prompt_text
        self.prompt_tokens = self.tokenizer.encode(prompt_text)
        return self
    

    def get_prompt_token_probabilities(self):
        """
        Calculate the probability of each token in the sequence given its preceding context,
        using a single forward pass.
        
        Args:
            self: The Explorer object
        Returns:
            list: A list of probabilities for each token in the sequence
        """
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get the model's output in a single forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
        
        # Calculate probabilities for each position
        token_probabilities = []
        
        # First token has no context, so we'll use None or some default
        token_probabilities.append(0.5)
        
        # For each position after the first
        for pos in range(len(self.prompt_tokens) - 1):
            # The logits at position 'pos' predict the token at position 'pos+1'
            position_logits = logits[pos]
            position_probs = torch.softmax(position_logits, dim=-1)
            
            # Get probability of the actual next token
            next_token_id = self.prompt_tokens[pos + 1]
            next_token_prob = position_probs[next_token_id].item()
            
            token_probabilities.append(next_token_prob)
        return token_probabilities
    
    def get_prompt_token_normalized_entropies(self):
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get the model's output in a single forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
        
        # Calculate normalized entropy for each position
        normalized_entropies = []
        
        # First token has no context, so we'll use None or some default
        normalized_entropies.append(0.5)
        
        # For each position after the first
        for pos in range(len(self.prompt_tokens) - 1):
            # The logits at position 'pos' predict the token at position 'pos+1'
            position_logits = logits[pos]
            position_probs = torch.softmax(position_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            # We filter out zeros to avoid log(0) issues
            probs_np = position_probs.cpu().numpy()
            non_zero_probs = probs_np[probs_np > 0]
            entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
            
            # Normalize by maximum possible entropy (log2 of vocabulary size)
            max_entropy = np.log2(len(position_probs))
            normalized_entropy = entropy / max_entropy
            
            normalized_entropies.append(normalized_entropy)
        
        return normalized_entropies


    def get_prompt(self):
        """
        Get the current prompt text.
        
        Returns:
            The current prompt text
        """
        return self.prompt_text
    
    def get_prompt_tokens(self):
        """
        Get the current encoded prompt tokens.
        
        Returns:
            List of token ids representing the current prompt
        """
        return self.prompt_tokens
    
    def get_prompt_tokens_strings(self):
        """
        Get the current prompt tokens as a string.
        """
        return [self.tokenizer.decode(token) for token in self.prompt_tokens]
    
    def pop_token(self):
        """
        NOTE: Need to handle the guide in this case.
        Remove and return the last token from the prompt tokens.
        If the prompt is empty, return None.
        
        Returns:
            The removed token id, or None if prompt was empty
        """
        if not self.prompt_tokens:
            return None
            
        # Pop last token and update prompt text
        last_token = self.prompt_tokens.pop()
        self.prompt_text = self.tokenizer.decode(self.prompt_tokens)
        
        return last_token
    
    def append_token(self, token_id):
        """
        Append a token to the current prompt tokens and update prompt text.
        
        Args:
            token_id: The token id to append
        """
        # Add token to prompt tokens
        self.prompt_tokens.append(token_id)
        
        # Update prompt text to match new tokens
        self.prompt_text = self.tokenizer.decode(self.prompt_tokens)
        if self.guide is not None:
            self.guide.advance(token_id)
        
        return self
    
    def guide_is_finished(self):
        if self.guide is not None:
            return self.guide.is_finished()
        return False
    
    def get_top_n_tokens(self, n=5, search=""):
        #if self.guide_is_finished():
        #    return []
        """
        Get the top n most likely next tokens given the current prompt.
        Optionally filter tokens by a search string.
        
        Args:
            n: Number of top tokens to return (default 5)
            search: Optional string to filter tokens (default "")
            
        Returns:
            List of dicts containing token info and probabilities, sorted by probability
        """
        # Get model output for the encoded prompt
        with torch.no_grad():
            outputs = self.model(torch.tensor([self.prompt_tokens]).to(self.device))
            
        # Get logits for the next token
        next_token_logits = outputs.logits[0, -1, :]
        
        # Get probabilities using softmax
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)

        if self.guide is not None:
            allowed_tokens = self.guide.get_tokens()
            allowed_tokens_mask = torch.zeros(len(next_token_probs), device=next_token_logits.device)
            allowed_tokens_mask[allowed_tokens] = 1.0
            next_token_probs =  next_token_probs * allowed_tokens_mask
            # renormalize the probabilities
            next_token_probs = next_token_probs / next_token_probs.sum()
        if search:
            # Filter tokens that contain the search string
            matching_tokens = []
            for idx, prob in enumerate(next_token_probs):
                token = self.tokenizer.decode(idx)
                if search.lower() in token.lower():
                    matching_tokens.append({
                        "token_id": idx,
                        "token": token,
                        "probability": prob.item()
                    })
            
            # Sort by probability and take top n
            matching_tokens.sort(key=lambda x: x["probability"], reverse=True)
            if self.guide is not None:
                # make sure that the token id is in the allowed tokens
                matching_tokens = [token for token in matching_tokens if token["token_id"] in allowed_tokens]
            return matching_tokens[:n]
        else:
            # Original behavior for no search string
            top_probs, top_indices = torch.topk(next_token_probs, n)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode(idx)
                results.append({
                    "token": token,
                    "token_id": idx.item(),
                    "probability": prob.item()
                })
            if self.guide is not None:
                # make sure that the token id is in the allowed tokens
                results = [token for token in results if token["token_id"] in allowed_tokens]
            return results

"""
Attempting to replicate the basic api of outlines-core, but
we're going to try to reduce the memory footprint and make it more efficient.

"""

# Example usage
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")  
    # test the RegexGuide
    guide = RegexGuide(r'a{1,5}', tokenizer)
    print("Tokens:", guide.get_tokens())
    guide.advance('a')
    print("Tokens:", guide.get_tokens())
    guide.advance('a')
    print("Tokens:", guide.get_tokens())
    explorer = Explorer()
    explorer.set_prompt("Once upon a time, there was a")
    
    print("Prompt:", explorer.get_prompt())
    print("Encoded prompt:", explorer.get_prompt_tokens())
    print("-----")
    print("Top tokens:", explorer.get_top_n_tokens())
    print("-----")
    print("Filtered tokens:", explorer.get_top_n_tokens(search="man"))
    print("-----")
    print("Appending token:", explorer.get_top_n_tokens(search="man")[0])
    explorer.append_token(explorer.get_top_n_tokens(search="man")[0]["token_id"])
    print("-----")
    print("Prompt:", explorer.get_prompt())
    print("Encoded prompt:", explorer.get_prompt_tokens())
    print("-----")
    print("Popping token:", explorer.pop_token())
    print("-----")
    print("Prompt:", explorer.get_prompt()) 
    print("Token probabilities:", explorer.get_prompt_token_probabilities())
    print("-----")
    print("Token entropies:", explorer.get_prompt_token_normalized_entropies())
    explorer.set_guide(r'a{1,5}')
    print("-----")
    print("Top tokens:", explorer.get_top_n_tokens())
    print("-----")
    print("Guide is finished:", explorer.guide.is_finished())

