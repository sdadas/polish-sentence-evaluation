from pathlib import Path

from data import Corpus
from embeddings.laser_embeddings import LaserEmbedding
from embeddings.word_embeddings import RandomEmbedding
from methods.use import USEEmbedding

if __name__ == '__main__':
    corpus = Corpus(1)
    path: Path = Path("resources/sample.txt")
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            corpus.add_sample(line.strip())
    print(len(corpus))
    emb = USEEmbedding()
    emb.embed_corpus(corpus, "use.npy")