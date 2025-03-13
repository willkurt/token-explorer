from time import monotonic
from src.explorer import Explorer
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Digits, Footer, Header, Static, Log, Markdown, DataTable

EXAMPLE_PROMPT = "Once upon a time, there was a"
# this will be replaced with a function to get the tokenization strings.
token_strs = EXAMPLE_PROMPT.split(" ")
TOKENS_TO_SHOW = 30
class TokenExplorer(App):
    """Main application class."""
    def __init__(self):
        super().__init__()
        self.explorer = Explorer()
        self.explorer.set_prompt(EXAMPLE_PROMPT)
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
    
    def _top_tokens_to_rows(self, tokens):
        return [("token_id", "token", "rel prob")] + [
            (token["token_id"], token["token"], token["probability"])
            for token in tokens
        ]
        
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Markdown(id="results")
            yield DataTable(id="table")
        yield Footer()
    
    def on_mount(self) -> None:
        self.query_one("#results", Markdown).update(f"Prompt: {self.explorer.get_prompt()}")
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"
    
    def on_key(self, event) -> None:
        """Handle key events on the data table."""
        table = self.query_one(DataTable)
        
        # Check if right arrow key was pressed
        if event.key == "right":
            # Get the currently selected row index
            row_index = table.cursor_row
            
            if row_index is not None:
                self.explorer.append_token(self.rows[row_index+1][0])
                self.query_one("#results", Markdown).update(f"Prompt: {self.explorer.get_prompt()}")
                self.rows = self._top_tokens_to_rows(
                    self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
                    )
                table.clear()
                #table.add_columns(*self.rows[0])
                table.add_rows(self.rows[1:])
        elif event.key == "left":
            self.explorer.pop_token()
            self.rows = self._top_tokens_to_rows(
                self.explorer.get_top_n_tokens(n=10)
                )
            table.clear()
            table.add_rows(self.rows[1:])
            self.query_one("#results", Markdown).update(f"Prompt: {self.explorer.get_prompt()}")

if __name__ == "__main__":
    app = TokenExplorer()
    app.run()