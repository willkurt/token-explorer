from ast import literal_eval
from itertools import cycle
from src.explorer import Explorer
from src.utils import entropy_to_color, probability_to_color
from textual.app import App, ComposeResult, Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static, DataTable
from textwrap import dedent
import sys
import os
import argparse
import tomli 
from datetime import datetime
import re

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

    BINDINGS = [("e", "change_display_mode", "Mode"),
                ("left,h", "pop_token", "Back"),
                ("right,l", "append_token", "Add"),
                ("d", "add_prompt", "New"),
                ("a", "remove_prompt", "Del"),
                ("w", "increment_prompt", "Next"),
                ("s", "decrement_prompt", "Prev"),
                ("x", "save_prompt", "Save"),
                ("j", "select_next", "Down"),
                ("k", "select_prev", "Up"),
                ("r", "toggle_struct", "Toggle struct"),
                ("R", "next_struct", "Next struct")

                ]
    
    
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
        self.selected_row = 0  # Track currently selected token row
        self.regex_structs = self._get_regex_structs()
        # this is the position of the stuct in the prompt
        self.struct_index = None
        # this is the position of the struct in the regex_structs list
        self.current_struct_index = 0
    
    def _get_regex_structs(self):
        try:
            struct_files = []
            # Get all files in struct directory
            for file in os.listdir("struct"):
                if file.endswith(".txt"):
                    file_path = os.path.join("struct", file)
                    try:
                        with open(file_path, "r") as f:
                            # Get first line and strip whitespace
                            regex = f.readline().strip()
                            # Remove file extension and add tuple
                            name = os.path.splitext(file)[0]
                            struct_files.append((name, str(literal_eval(regex))))
                    except:
                        # Skip files that can't be read
                        continue
            return struct_files
        except FileNotFoundError:
            return []

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
        print("Rows:", self.rows)
        table.clear()
        table.add_rows(self.rows[1:])
        # Reset cursor to top
        self.selected_row = 0
        table.move_cursor(row=self.selected_row)
        self.query_one("#results", Static).update(self._render_prompt())
    

    def _render_structure_section(self):
        struct_section = ""
        if self.explorer.guide_is_finished():
            struct_section = f"[on red]{self.regex_structs[self.current_struct_index][0]}[/on]"
        elif self.struct_index is not None:
            struct_section = f"[on green]{self.regex_structs[self.current_struct_index][0]}[/on]"

        else:
            struct_section = f"[on grey]{self.regex_structs[self.current_struct_index][0]}[/on]"
        return struct_section
    
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
[bold]Struct[/bold] {self._render_structure_section()}
""")
    
    def on_mount(self) -> None:
        self.query_one("#results", Static).update(self._render_prompt())
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"

    def action_next_struct(self):
        self.current_struct_index = (self.current_struct_index + 1) % len(self.regex_structs)
        self.query_one("#results", Static).update(self._render_prompt())

    def action_toggle_struct(self):
        if self.struct_index is None:
            # this is the theoretical index of the first
            # structure token when structured gen is activated
            # even though that token *doesn't* exist yet.
            # this track to help with backtracking.
            self.struct_index = len(self.explorer.get_prompt_tokens())
            self.explorer.set_guide(self.regex_structs[self.current_struct_index][1])
        else:
            self.struct_index = None
            self.explorer.clear_guide()
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()
        
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

    def action_save_prompt(self):
        with open(f"prompts/prompt_{self.prompt_index}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as f:
            f.write(self.explorer.get_prompt())

    def action_select_next(self):
        """Move selection down one row"""
        if self.selected_row < len(self.rows) - 2:  # -2 for header row
            self.selected_row += 1
            table = self.query_one(DataTable)
            table.move_cursor(row=self.selected_row)
            
    def action_select_prev(self):
        """Move selection up one row"""
        if self.selected_row > 0:
            self.selected_row -= 1
            table = self.query_one(DataTable)
            table.move_cursor(row=self.selected_row)

    def action_append_token(self):
        """Append currently selected token"""
        if self.explorer.guide_is_finished():
            return None
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            self.explorer.append_token(self.rows[table.cursor_row+1][0])
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self._refresh_table()  # This will reset cursor position

    def action_pop_token(self):
        if len(self.explorer.get_prompt_tokens()) > 1:
            self.explorer.pop_token()
            if self.explorer.guide is not None:
                self.explorer.clear_guide()
                #  need to add logic for backtracking the guide
                self.explorer.set_guide(self.regex_structs[self.current_struct_index][1]
                                        ,ff_from=self.struct_index)
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
            

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
