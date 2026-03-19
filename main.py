import numpy as np
from dados import create_vocab, sentence_to_ids
from embeddings import create_embedding_table
from inference import generate_sequence

vocab = create_vocab()

vocab_size = len(vocab)
embedding_table = create_embedding_table(vocab_size, d_model=64)

sentence = "o banco bloqueou cartao"
x_ids = np.array(sentence_to_ids(sentence, vocab))

start_token = 0
eos_token = 3

output = generate_sequence(x_ids, embedding_table, vocab_size, start_token, eos_token)

print("Saída:", output)