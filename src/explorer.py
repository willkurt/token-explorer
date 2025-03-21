"""
This code is used to process an LLM one token at at time.

The Explorer class manages the prompt internally and handles all interactions with the LLM.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

class Explorer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", use_bf16=False):
        """
        Initialize the Explorer with a model name.
        
        Args:
            model_name: Name of the model to load (default "Qwen/Qwen2.5-0.5B")
            use_bf16: Whether to load model in bf16 precision (default False)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load model with bf16 if specified
        if use_bf16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Auto select device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        
        # Initialize with empty prompt
        self.prompt_text = ""
        self.prompt_tokens = []
    
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
            
            # Convert to float32 for probability calculation
            position_probs_float = position_probs.to(torch.float32)
            
            # Get probability of the actual next token
            next_token_id = self.prompt_tokens[pos + 1]
            next_token_prob = position_probs_float[next_token_id].item()
            
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
            # Convert to float32 for entropy calculation
            position_probs_float = position_probs.to(torch.float32)
            # We filter out zeros to avoid log(0) issues
            probs_np = position_probs_float.cpu().numpy()
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
        Returns a list of strings with special tokens (newline, tab, etc.) made visible.
        """
        return [self._format_special_token(self.tokenizer.decode(token)) for token in self.prompt_tokens]
    
    def pop_token(self):
        """
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
        
        return self
    
    def _format_special_token(self, token):
        """
        Format special/invisible tokens into readable strings.
        
        Args:
            token: The token string to format
        Returns:
            Formatted string with special tokens made visible
        """
        # Common special tokens mapping
        special_tokens = {
            '\n': '\\n',  # newline
            '\t': '\\t',  # tab
            '\r': '\\r',  # carriage return
            ' ': '\\s',   # space
        }
        
        # If token is in special_tokens, return its visible representation
        if token in special_tokens:
            return special_tokens[token]
        return token

    def get_top_n_tokens(self, n=5, search=""):
        """
        Get the top n most likely next tokens given the current prompt.
        Optionally filter tokens by a search string.
        
        Args:
            n: Number of top tokens to return (default 5)
            search: Optional string to filter tokens (default "")
            
        Returns:
            List of dicts containing token info and probabilities, sorted by probability
        """
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get model output with hidden states
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            
        # Get logits and hidden states
        next_token_logits = outputs.logits[0, -1, :]
        hidden_states = outputs.hidden_states  # Tuple of tensors (num_layers + 1, batch, seq_len, hidden_size)
        
        # Get probabilities for final layer using softmax
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0).to(torch.float32)
        
        # Calculate per-layer probabilities
        # Skip first element as it's the embeddings, not a layer output
        layer_probs = []
        for layer_output in hidden_states[1:]:
            # Get last token's hidden state
            last_hidden = layer_output[0, -1, :]
            # Project to vocab size using model's lm_head
            layer_logits = self.model.lm_head(last_hidden.unsqueeze(0)).squeeze(0)
            # Get probabilities
            layer_probs.append(torch.nn.functional.softmax(layer_logits, dim=0).to(torch.float32))
        
        if search:
            # Filter tokens that contain the search string
            matching_tokens = []
            for idx, prob in enumerate(next_token_probs):
                token = self._format_special_token(self.tokenizer.decode(idx))
                if search.lower() in token.lower():
                    matching_tokens.append({
                        "token_id": idx,
                        "token": token,
                        "probability": prob.item(),
                    "layer_probs": [layer_prob[idx].item() for layer_prob in layer_probs]
                    })
            
            # Sort by probability and take top n
            matching_tokens.sort(key=lambda x: x["probability"], reverse=True)
            return matching_tokens[:n]
        else:
            # Original behavior for no search string
            top_probs, top_indices = torch.topk(next_token_probs, n)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self._format_special_token(self.tokenizer.decode(idx))
                results.append({
                    "token": token,
                    "token_id": idx.item(),
                    "probability": prob.item(),
                    "layer_probs": [layer_prob[idx].item() for layer_prob in layer_probs]
                })
                
            return results


# Example usage
if __name__ == "__main__":
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
