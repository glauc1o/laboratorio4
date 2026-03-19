import numpy as np
from encoder import encoder
from decoder import decoder
from embeddings import get_embeddings

def generate_sequence(x_ids, embedding_table, vocab_size, start_token, eos_token, max_len=10):
    X = get_embeddings(x_ids, embedding_table)
    Z = encoder(X)

    y_ids = [start_token]

    for _ in range(max_len):
        Y = get_embeddings(np.array(y_ids), embedding_table)
        Y = np.expand_dims(Y, axis=0)

        probs = decoder(Y, Z, vocab_size)

        next_token = np.argmax(probs[0, -1])

        y_ids.append(next_token)

        if next_token == eos_token:
            break

    return y_ids