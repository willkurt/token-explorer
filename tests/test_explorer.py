from src.explorer import Explorer

def test_get_prompt_token_probabilities():
    explorer = Explorer()
    explorer.set_prompt("Hello, world")
    probabilities = explorer.get_prompt_token_probabilities()
    assert probabilities[0] == 0.5 # first token has no context
    assert len(probabilities) == len(explorer.prompt_tokens)

def test_get_top_n_tokens():
    explorer = Explorer()
    explorer.set_prompt("Hello, world")
    tokens = explorer.get_top_n_tokens(n=5)
    assert len(tokens) == 5
    assert tokens[0]["token"] == "!"

def test_guide():
    explorer = Explorer()
    explorer.set_prompt("Hello, world")
    explorer.set_guide("ba+")
    tokens = explorer.get_top_n_tokens(n=5)
    assert len(tokens) == 2 # actually only 2 valid tokens
    assert tokens[0]["token"][0] == "b"

def test_guide_append_token():
    explorer = Explorer()
    tokenizer = explorer.tokenizer
    explorer.set_prompt("Hello, world")
    explorer.set_guide("abc")
    tokens = explorer.get_top_n_tokens(n=5)
    assert tokens[0]["token"][0] == "a"
    explorer.append_token(tokenizer.encode("a")[0])
    tokens = explorer.get_top_n_tokens(n=5)
    assert tokens[0]["token"][0] == "b"