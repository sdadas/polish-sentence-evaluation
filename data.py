from typing import List

from analyzer import PolishAnalyzer


class Sentence(object):

    def __init__(self, raw: str, tokenized: List[str], lemmatized: List[str]):
        self.raw = raw
        self.tokens = tokenized
        self.lemmas = lemmatized


class Corpus(object):

    def __init__(self):
        self.analyzer = PolishAnalyzer()
        self.sentences: List[Sentence] = []

    def add(self, sentence: str):
        tokens, lemmas = self.analyzer.analyze(sentence)
        result = Sentence(sentence, tokens, lemmas)
        self.sentences.append(result)
        return result


if __name__ == '__main__':
    corpus = Corpus()
    sent = corpus.add("Pchnąć jeża w tę łódź i ośm skrzyń fig!")
    print(sent.tokens)
    print(sent.lemmas)