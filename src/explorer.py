"""
This code is used to process an LLM one token at at time.

The Explorer class manages the prompt internally and handles all interactions with the LLM.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
        # Get model output for the encoded prompt
        with torch.no_grad():
            outputs = self.model(torch.tensor([self.prompt_tokens]))
            
        # Get logits for the next token
        next_token_logits = outputs.logits[0, -1, :]
        
        # Get probabilities using softmax
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        
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