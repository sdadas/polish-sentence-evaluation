import base64
import hashlib
import re
import json
import subprocess
from typing import TextIO, Callable, Optional, Union

from dataclasses import dataclass, asdict


def wcl(file_path):
    p = subprocess.Popen(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


class Sentence:

    non_alphanum = re.compile("[\W_]+", re.UNICODE)

    def __init__(self, text: str):
        self.__text = text
        self.__norm = None

    def text(self) -> str:
        return self.__text

    def norm(self) -> str:
        if not self.__norm:
            self.__norm = self.create_norm(self.__text)
        return self.__norm

    def words(self):
        norm = self.norm()
        return set(norm.split())

    def set_text(self, text: str):
        self.__text = text
        self.__norm = None

    @staticmethod
    def create_norm(text: str):
        norm = Sentence.non_alphanum.sub(" ", text)
        return re.sub(r"\s+", " ", norm).lower()


@dataclass
class DetailedScore:
    label: Optional[str] = None
    similarity: Optional[float] = None
    levenshtein: Optional[float] = None


class SentencePair:

    def __init__(self, sent1: str, sent2: str, score,  clean=True, label=None, method=None, source=None):
        self.sent1 = Sentence(self.__clean(sent1) if clean else sent1)
        self.sent2 = Sentence(self.__clean(sent2) if clean else sent2)
        self.score: Union[float, DetailedScore] = score
        self.label= label
        self.method = method
        self.source = source

    def __clean(self, sent: str):
        eos = sent[-1]
        if eos in (".", "?", "!"): sent = sent[:-1]
        else: eos = ""
        sent = sent.strip("\' \n\r-—–“”„“`").replace("\t", " ")
        return sent + eos

    def min_length(self):
        return min(len(self.sent1.text()), len(self.sent2.text()))

    def max_length(self):
        return max(len(self.sent1.text()), len(self.sent2.text()))

    def similar_norms(self) -> bool:
        norm1 = self.sent1.norm()
        norm2 = self.sent2.norm()
        if norm1 == norm2: return True
        elif norm1.startswith(norm2) or norm2.startswith(norm1): return True
        elif norm1.endswith(norm2) or norm2.endswith(norm1): return True
        return False

    def word_overlap(self) -> float:
        words1 = self.sent1.words()
        words2 = self.sent2.words()
        if len(words1) == 0 or len(words2) == 0: return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def text_hash(self):
        str_min = min(self.sent1.text(), self.sent2.text())
        str_max = max(self.sent1.text(), self.sent2.text())
        hash = hashlib.sha512((str_min + str_max).encode("utf-8")).digest()
        return base64.b64encode(hash).decode("ascii")

    def to_dict(self):
        res = {"sent1": self.sent1.text(), "sent2": self.sent2.text()}
        res["score"] = asdict(self.score) if isinstance(self.score, DetailedScore) else self.score
        res["overlap"] = self.word_overlap()
        res["hash"] = self.text_hash()
        if self.method: res["method"] = self.method
        if self.source: res["source"] = self.source
        if self.label: res["label"] = self.label
        return res

    def to_tuple(self):
        return [self.sent1.text(), self.sent2.text(), self.score, self.word_overlap()]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_tsv(self) -> str:
        return "\t".join(str(val) for val in self.to_tuple())

    def formatted(self, format):
        return {"json": self.to_json, "tsv": self.to_tsv}[format]()


class FormattedWriter(object):

    def __init__(self, outfile: TextIO, format: str, add_newline: bool=True):
        self.outfile = outfile
        self.format = format
        self.write_func: Callable = self.__write_json if format == "json" else self.__write_default
        self.add_newline = add_newline
        self.idx = 0
        self.start()

    def start(self):
        if self.format == "json":
            self.outfile.write("[")

    def write(self, line: str):
        self.write_func(line)
        if self.add_newline:
            self.outfile.write("\n")
        self.outfile.flush()
        self.idx += 1

    def __write_default(self, line: str):
        self.outfile.write(line)

    def __write_json(self, line: str):
        if self.idx > 0: self.outfile.write(",")
        self.outfile.write(line)

    def finish(self):
        if self.format == "json":
            self.outfile.write("]")
