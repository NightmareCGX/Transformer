def test_placeholder_forward():
    from transformer.model import PlaceholderModel

    m = PlaceholderModel(vocab_size=1000)
    out = m.forward([1, 2, 3])
    assert out == [1, 2, 3]
