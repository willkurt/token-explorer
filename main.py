from itertools import cycle
from src.explorer import Explorer
from src.utils import entropy_to_color, probability_to_color
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static, DataTable
from textwrap import dedent
import sys
import argparse
import tomli 
from datetime import datetime

# Replace the constants with config
def load_config():
    try:
        with open("config.toml", "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        print("Config file not found, using default values")
        return {
            "model": "Qwen/Qwen2.5-0.5B",
            "example_prompt": "Once upon a time, there was a",
            "tokens_to_show": 30,
            "max_prompts": 9
        }

config = load_config()
MODEL_NAME = config["model"]["name"]
EXAMPLE_PROMPT = config["prompt"]["example_prompt"]
TOKENS_TO_SHOW = config["display"]["tokens_to_show"]
MAX_PROMPTS = config["prompt"]["max_prompts"]

class TokenExplorer(App):
    """Main application class."""

    display_modes = cycle(["prompt", "prob", "entropy"])
    display_mode = reactive(next(display_modes))

    BINDINGS = [("e", "change_display_mode", "Change display mode"),
                ("left", "pop_token", "Pop token"),
                ("right", "append_token", "Append token"),
                ("d", "add_prompt", "Add prompt"),
                ("a", "remove_prompt", "Remove prompt"),
                ("w", "increment_prompt", "Increment prompt"),
                ("s", "decrement_prompt", "Decrement prompt"),
                ("x", "save_prompt", "Save prompt")]
    
    def __init__(self, prompt=EXAMPLE_PROMPT):
        super().__init__()
        # Add support for multiple prompts.
        self.prompts = [prompt]
        self.prompt_index = 0
        self.explorer = Explorer(MODEL_NAME)
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
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
        table.clear()
        table.add_rows(self.rows[1:])
        self.query_one("#results", Static).update(self._render_prompt())
            
    def _render_prompt(self):
        if self.display_mode == "entropy":
            entropy_legend = "".join([
                f"[on {entropy_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token entropy:[/bold]{entropy_legend}"
            token_entropies = self.explorer.get_prompt_token_normalized_entropies()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {entropy_to_color(entropy)}]{token}[/on]" for token, entropy in zip(token_strings, token_entropies))
        elif self.display_mode == "prob":
            prob_legend = "".join([
                f"[on {probability_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token prob:[/bold]{prob_legend}"
            token_probs = self.explorer.get_prompt_token_probabilities()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {probability_to_color(prob)}]{token}[/on]" for token, prob in zip(token_strings, token_probs))
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = ""
        return dedent(f"""
{prompt_text}





{prompt_legend}
[bold]Prompt[/bold] {self.prompt_index+1}/{len(self.prompts)} tokens: {len(self.explorer.prompt_tokens)}
""")
    
    def on_mount(self) -> None:
        self.query_one("#results", Static).update(self._render_prompt())
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"
    
    def action_add_prompt(self):
        if len(self.prompts) < MAX_PROMPTS:
            self.prompts.append(self.explorer.get_prompt())
            self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
            self.explorer.set_prompt(self.prompts[self.prompt_index])
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()

    def action_remove_prompt(self):
        if len(self.prompts) > 1:
            self.prompts.pop(self.prompt_index)
            self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
            self.explorer.set_prompt(self.prompts[self.prompt_index])
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
    
    def action_increment_prompt(self):
        self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    def action_decrement_prompt(self):
        self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    def action_change_display_mode(self):
        self.display_mode = next(self.display_modes)
        self.query_one("#results", Static).update(self._render_prompt())

    def action_pop_token(self):
        if len(self.explorer.get_prompt_tokens()) > 1:
            self.explorer.pop_token()
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
            
    def action_save_prompt(self):
        with open(f"prompts/prompt_{self.prompt_index}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as f:
            f.write(self.explorer.get_prompt())

    def action_append_token(self):
        table = self.query_one(DataTable)
        row_index = table.cursor_row
        
        if row_index is not None:
            self.explorer.append_token(self.rows[row_index+1][0])
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self.query_one("#results", Static).update(self._render_prompt())
            self.rows = self._top_tokens_to_rows(
                self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
                )
            table.clear()
            table.add_rows(self.rows[1:])
        self.query_one("#results", Static).update(self._render_prompt())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Token Explorer Application')
    parser.add_argument('--input', '-i', type=str, help='Path to input text file')
    args = parser.parse_args()

    prompt = EXAMPLE_PROMPT
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
    app = TokenExplorer(prompt)
    app.run()

