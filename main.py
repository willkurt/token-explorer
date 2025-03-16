from itertools import cycle
from time import monotonic
from src.explorer import Explorer
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Digits, Footer, Header, Static, Log, Markdown, DataTable
from textwrap import dedent
import sys
import argparse

EXAMPLE_PROMPT = "Once upon a time, there was a"
# this will be replaced with a function to get the tokenization strings.
token_strs = EXAMPLE_PROMPT.split(" ")
TOKENS_TO_SHOW = 30

def probability_to_color(probability, alpha=1.0):
    """
    Maps a probability value (0.0-1.0) to a color on a blue-red scale.
    Blue represents high probability (1.0)
    Red represents low probability (0.0)
    
    Args:
        probability (float): Probability value between 0.0 and 1.0
        alpha (float, optional): Alpha/opacity value between 0.0 and 1.0. Defaults to 1.0.
    
    Returns:
        str: RGBA color string (format: 'rgba(r, g, b, a)')
    """
    # Ensure probability is in valid range
    probability = max(0, min(1, probability))
    
    # Red component (high when probability is low)
    red = int(255 * (1 - probability))
    
    # Blue component (high when probability is high)
    blue = int(255 * probability)
    
    # Green component (kept at 0 for a cleaner red-blue gradient)
    green = 0
    
    # Return rgba string
    return f"rgba({red}, {green}, {blue}, {alpha})"

def entropy_to_color(entropy, alpha=1.0):
    """
    Maps a normalized entropy value (0.0-1.0) to a grayscale color.
    White (255,255,255) represents highest entropy (1.0)
    Black (0,0,0) represents lowest entropy (0.0)
    
    Args:
        entropy (float): Normalized entropy value between 0.0 and 1.0
        alpha (float, optional): Alpha/opacity value between 0.0 and 1.0. Defaults to 1.0.
    
    Returns:
        str: RGBA color string (format: 'rgba(r, g, b, a)')
    """
    # Ensure entropy is in valid range
    entropy = max(0, min(1, entropy))
    
    # For grayscale, all RGB components have the same value
    # Higher entropy = lighter color (closer to white)
    value = int(255 * entropy)
    
    # Return rgba string
    return f"rgba({value}, {value}, {value}, {alpha})"

class TokenExplorer(App):
    """Main application class."""
    display_mode = reactive("prompt")
    display_modes = cycle(["prompt", "prob", "entropy"])

    def __init__(self, prompt=EXAMPLE_PROMPT):
        super().__init__()
        # Add support for multiple prompts.
        self.prompts = [prompt]
        self.explorer = Explorer()
        self.explorer.set_prompt(prompt)
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
    
    def _top_tokens_to_rows(self, tokens):
        return [("token_id", "token", "prob")] + [
            (token["token_id"], token["token"], token["probability"])
            for token in tokens
        ]
        
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="results")
            yield DataTable(id="table")
        yield Footer()

    def _refresh_table(self):
        table = self.query_one(DataTable)
        table.clear()
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"
            
    def _render_prompt(self):
        if self.display_mode == "entropy":
            entropy_legend = "".join([
                f"[on {entropy_to_color(i/10)}] {i/10:.2f} "
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token entropy:[/bold]{entropy_legend}"
            token_entropies = self.explorer.get_prompt_token_normalized_entropies()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {entropy_to_color(entropy)}]{token}" for token, entropy in zip(token_strings, token_entropies))
        elif self.display_mode == "prob":
            prob_legend = "".join([
                f"[on {probability_to_color(i/10)}] {i/10:.2f} "
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token prob:[/bold]{prob_legend}"
            token_probs = self.explorer.get_prompt_token_probabilities()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {probability_to_color(prob)}]{token}" for token, prob in zip(token_strings, token_probs))
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = ""
        return dedent(f"""
{prompt_text}





{prompt_legend}
[bold]Prompt[/bold] 1/1 tokens: {len(self.explorer.prompt_tokens)}
""")
    
    def on_mount(self) -> None:
        self.query_one("#results", Static).update(self._render_prompt())
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"
    

    # Pretty sure I can refactor this to use
    # a better key binding setup.
    def on_key(self, event) -> None:
        """Handle key events on the data table."""
        table = self.query_one(DataTable)
        
        # Check if right arrow key was pressed
        if event.key == "right":
            # Get the currently selected row index
            row_index = table.cursor_row
            
            if row_index is not None:
                self.explorer.append_token(self.rows[row_index+1][0])
                self.query_one("#results", Static).update(self._render_prompt())
                self.rows = self._top_tokens_to_rows(
                    self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
                    )
                table.clear()
                #table.add_columns(*self.rows[0])
                table.add_rows(self.rows[1:])
        elif event.key == "left":
            self.explorer.pop_token()
            self.rows = self._top_tokens_to_rows(
                self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
                )
            table.clear()
            table.add_rows(self.rows[1:])
            self.query_one("#results", Static).update(self._render_prompt())
        elif event.key == "e":
            # this is enough?
            self.display_mode = next(self.display_modes)
            self.query_one("#results", Static).update(self._render_prompt())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Token Explorer Application')
    parser.add_argument('--input', '-i', type=str, help='Path to input text file')
    args = parser.parse_args()

    prompt = None
    if args.input:
        try:
            with open(args.input, 'r') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Could not find input file '{args.input}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    print(prompt)
    app = TokenExplorer(prompt)
    app.run()

