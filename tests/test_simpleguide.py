from src.simpleguide import SimpleGuide
from transformers import AutoTokenizer

def test_get_tokens():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    # test a basic regex
    guide = SimpleGuide("a+", tokenizer)
    decoded_tokens = [tokenizer.decode(token) for token in guide.get_tokens()]
    for token in decoded_tokens:
        for i in range(len(token)):
            assert token[i] == "a"

def test_advance():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide("abbbc", tokenizer)
    guide.advance(tokenizer.encode("a")[0])
    decoded_tokens = [tokenizer.decode(token) for token in guide.get_tokens()]
    for token in decoded_tokens:
        assert token[0] == "b"

def test_is_finished_single_finish():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide("abc", tokenizer)
    guide.advance(tokenizer.encode("a")[0])
    assert not guide.is_finished()
    guide.advance(tokenizer.encode("b")[0])
    assert not guide.is_finished()
    guide.advance(tokenizer.encode("c")[0])
    assert guide.is_finished()

def test_is_finished_multiple_finish():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide("abc{1,2}", tokenizer)
    guide.advance(tokenizer.encode("a")[0])
    assert not guide.is_finished()
    guide.advance(tokenizer.encode("b")[0])
    assert not guide.is_finished()
    guide.advance(tokenizer.encode("c")[0])
    assert guide.is_finished()
    guide.advance(tokenizer.encode("c")[0])
    assert guide.is_finished()

def test_is_dead():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide("abc", tokenizer)
    guide.advance(tokenizer.encode("a")[0])
    assert not guide.is_dead()
    guide.advance(tokenizer.encode("b")[0])
    assert not guide.is_dead()
    guide.advance(tokenizer.encode("c")[0])
    assert guide.is_dead()

def test_is_dead_multiple_finish():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide("abc{1,2}", tokenizer)
    guide.advance(tokenizer.encode("a")[0])
    assert not guide.is_dead()
    guide.advance(tokenizer.encode("b")[0])
    assert not guide.is_dead()
    guide.advance(tokenizer.encode("c")[0])
    assert not guide.is_dead() and guide.is_finished()
    guide.advance(tokenizer.encode("c")[0])
    assert guide.is_dead()

def test_spaces():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    guide = SimpleGuide(" ab", tokenizer, no_cache=True)
    assert any([cc.accepts(' ') for cc in guide.fsm.map[0].keys()])
    assert tokenizer.encode(" ")[0] in guide.state_token_map[0]

    
