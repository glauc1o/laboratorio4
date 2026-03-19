
def create_vocab():
    vocab = {
        "o": 0,
        "banco": 1,
        "bloqueou": 2,
        "cartao": 3
    }

def sentence_to_ids(sentence, vocab):
    tokens = sentence.split()
    return [vocab[token] for token in tokens]