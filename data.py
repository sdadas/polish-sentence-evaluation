from typing import List, Iterable, Union
from analyzer import PolishAnalyzer


class Sent(object):

    def __init__(self, raw: str, tokenized: List[str], lemmatized: List[str]):
        self.raw = raw
        self.tokens = tokenized
        self.lemmas = lemmatized


class Corpus(object):

    def __init__(self, sentences_per_sample: int=1):
        self.analyzer = PolishAnalyzer()
        self.samples: List[Iterable[Sent]] = []
        self.sentences_per_sample = sentences_per_sample

    def add_sample(self, sentences: Union[str, Iterable[str]]) -> Union[Sent, Iterable[Sent]]:
        results: List[Sent] = []
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences:
            tokens, lemmas = self.analyzer.analyze(sentence)
            result = Sent(sentence, tokens, lemmas)
            results.append(result)
        self.samples.append(results)
        return results if len(results) > 1 else results[0]

    def raw_texts(self) -> List[str]:
        res: List[str] = []
        for sample in self.samples:
            for sent in sample:
                res.append(sent.raw)
        return res

    def __len__(self):
        return len(self.samples)
