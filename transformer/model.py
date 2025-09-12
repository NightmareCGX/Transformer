class PlaceholderModel:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def forward(self, x):
        return x
