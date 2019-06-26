from pathlib import Path

from data import Corpus
from embeddings.word_embeddings import RandomEmbedding

if __name__ == '__main__':
    corpus = Corpus(1)
    path: Path = Path("resources/sample.txt")
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            corpus.add_sample(line.strip())
    print(len(corpus))
    emb = RandomEmbedding()
    emb.embed_corpus(corpus, "random.npy")