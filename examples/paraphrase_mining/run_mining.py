import logging
import os
import random
import zipfile
import json
from collections import Counter, defaultdict
from typing import List, TextIO, Dict
import fire
import requests
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
from tqdm import tqdm
from lsm import LSM
from examples.paraphrase_mining.common import SentencePair, wcl, DetailedScore, Sentence


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.INFO)


DATASETS = {
    "opensubtitles": ("OPUS-OpenSubtitles", "v2018"),
    "cc_matrix": ("OPUS-CCMatrix", "v1"),
    "wiki": ("OPUS-WikiMatrix", "v1"),
    "jrc": ("OPUS-JRC-Acquis", "v3.0")
}


class ParallelIndex:

    def __init__(self, from_lang: str, to_lang: str, cache_dir: str="cache"):
        self.db = None
        self.seq = None
        self.cache_dir = cache_dir
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.base_url = "https://object.pouta.csc.fi"
        self.counter = defaultdict(Counter)

    def add_collection(self, collection_name: str, version: str=""):
        logging.info("Adding collection %s", collection_name)
        if len(version) > 0: version += "/"
        url = f"{self.base_url}/{collection_name}/{version}moses/{self.from_lang}-{self.to_lang}.txt.zip"
        output_dir = os.path.join(self.cache_dir, collection_name)
        if os.path.exists(output_dir):
            logging.info("Collection already exists, skipping download...")
        else:
            os.makedirs(output_dir, exist_ok=False)
            self._download(url, output_dir)
        self._index_collection(collection_name, output_dir)

    def write_paraphrases(self, output_path: str, min_count: int=2):
        logging.info("Writing paraphrases to file %s", output_path)
        with open(output_path, "w", encoding="utf-8") as outfile:
            for key, val in self.counter.items():
                count = val["all"]
                sentence_count = len(val.keys()) - 1
                if count < min_count or sentence_count < 2:
                    continue
                outfile.write(json.dumps(self._create_output(key, val), ensure_ascii=False))
                outfile.write("\n")

    def _create_output(self, from_id, mapping):
        results = []
        for key, val in mapping.items():
            if key == "all": continue
            results.append({"sent": self._sentence_by_id(key), "count": val})
        return {"from_sentence": self._sentence_by_id(from_id), "to_sentences": results, "total": mapping["all"]}

    def _download(self, url: str, output_dir: str):
        logging.info("Downloading and extracting %s", url)
        file_name = url.split("/")[-1]
        file_path = os.path.join(output_dir, file_name)
        file_size = int(requests.head(url).headers["Content-Length"])
        response = requests.get(url, stream=True)
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=url.split('/')[-1])
        with open(file_path, "wb") as zip_file:
            for data in response.iter_content(chunk_size=1024):
                zip_file.write(data)
                progress.update(1024)
        progress.close()
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(output_dir)
        os.remove(file_path)

    def _index_collection(self, collection_name: str, output_dir: str):
        logging.info("Indexing collection %s", collection_name)
        self.db = LSM(os.path.join(self.cache_dir, collection_name, "index.ldb"))
        self.seq = self.db["__seq__"] if "__seq__" in self.db else 0
        files = os.listdir(output_dir)
        first_file = lambda ext: list(filter(lambda v: v.endswith(f".{ext}"), files))[0]
        from_path = os.path.join(output_dir, first_file(self.from_lang))
        to_path = os.path.join(output_dir, first_file(self.to_lang))
        batch_size = 1000
        total_lines = wcl(from_path)
        progress = tqdm(total=total_lines, unit="lines", unit_scale=True)
        with open(from_path, "r", encoding="utf-8") as from_file, open(to_path, "r", encoding="utf-8") as to_file:
            idx = 1
            self.db.begin()
            while True:
                if idx % batch_size == 0:
                    self.db.commit()
                    self.db.begin()
                from_line = from_file.readline()
                to_line = to_file.readline()
                if not from_line:
                    break
                self._index_pair(from_line.strip(), to_line.strip())
                idx += 1
                progress.update(1)
        self.db.commit()
        self.db["__seq__"] = self.seq
        progress.close()

    def _index_pair(self, from_sent: str, to_sent: str):
        from_id = self._index_sent(from_sent, self.from_lang)
        to_id = self._index_sent(to_sent, self.to_lang)
        self.counter[from_id]["all"] += 1
        self.counter[from_id][to_id] += 1

    def _index_sent(self, sent: str, lang: str):
        sent_key = f"{lang}:{sent.lower()}"
        if sent_key in self.db:
            result = self.db[sent_key]
            result = json.loads(result)
            return result
        else:
            self.seq += 1
            sent_id = self.seq
            result = {"id": sent_id, "sent": sent, "lang": lang}
            self.db[sent_key] = sent_id
            self.db[sent_id] = json.dumps(result)
            return sent_id

    def _sentence_by_id(self, id):
        result = self.db[id]
        result = json.loads(result)
        return result["sent"]


class PairScorer:

    def __init__(self):
        self._load_sentence_encoder()


    def _load_sentence_encoder(self, encoder_name="xlm-r-distilroberta-base-paraphrase-v1"):
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder.eval()
        self.encoder.half()

    def batch_compute_similarity(self, batch: List[SentencePair]):
        sentences = [pair.sent1.text() for pair in batch]
        sentences.extend([pair.sent2.text() for pair in batch])
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        emb1 = embeddings[:len(batch), :]
        emb2 = embeddings[len(batch):, :]
        cosine_scores = util.pytorch_cos_sim(emb1, emb2)
        for idx in range(len(batch)):
            pair = batch[idx]
            pair.score.similarity = cosine_scores[idx, idx].item()

    def batch_compute_levenshtein(self, batch: List[SentencePair]):
        for pair in batch:
            str1 = pair.sent1.text()
            str2 = pair.sent2.text()
            pair.score.levenshtein = fuzz.ratio(str1, str2) / 100.0


class CorpusPairExtractor:

    def __init__(self, source: str):
        self.scorer = PairScorer()
        self.source = source

    def extract(self, input_path: str, output_path: str):
        logging.info("Generating paraphrase pairs to file %s", output_path)
        total_lines = wcl(input_path)
        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line in tqdm(infile, total=total_lines, unit="lines", unit_scale=True):
                value = json.loads(line.strip())
                pairs = self._extract_pairs(value)
                self._write_pairs(pairs, outfile)

    def _write_pairs(self, pairs: List[SentencePair], outfile: TextIO):
        for pair in pairs:
            outfile.write(pair.to_json())
            outfile.write("\n")

    def _extract_pairs(self, value: Dict):
        source = value["from_sentence"]
        sentences = value["to_sentences"]
        accepted, rejected = set(), set()
        pairs = [SentencePair(source, sent["sent"], DetailedScore()) for sent in sentences]
        self.scorer.batch_compute_similarity(pairs)
        for pair in pairs:
            sent = pair.sent2.text()
            if pair.score.similarity > 0.7:
                accepted.add(sent)
            else:
                rejected.add(sent)
        accepted = self._remove_duplicates(accepted)
        return self._generate_pairs_from_accepted(accepted) if len(accepted) > 1 else []

    def _remove_duplicates(self, sentences):
        norms = set()
        sentences = [sent.replace("\\", " ").replace("/", " ") for sent in sentences]
        sentences = [sent for sent in sentences if len(sent.strip()) > 0]
        results = []
        for sent in sentences:
            norm = Sentence.create_norm(sent)
            if norm not in norms:
                results.append(sent)
                norms.add(norm)
        return results

    def _generate_pairs_from_accepted(self, accepted: List[str]) -> List[SentencePair]:
        indices = list(range(len(accepted)))
        random.shuffle(indices)
        idx = 0
        pairs: List[SentencePair] = []
        while idx < len(indices):
            second_idx = idx + 1 if idx < (len(indices)-1) else 0
            sent1 = accepted[idx]
            sent2 = accepted[second_idx]
            pairs.append(SentencePair(sent1, sent2, DetailedScore(), source=self.source))
            idx += 2
        self.scorer.batch_compute_similarity(pairs)
        self.scorer.batch_compute_levenshtein(pairs)
        return pairs


def run_mining(from_lang: str, to_lang: str, dataset: str):
    assert dataset in DATASETS.keys(), f"dataset should be on of {', '.join(DATASETS.keys())}"
    c_name, c_version = DATASETS.get(dataset)
    index = ParallelIndex(from_lang, to_lang)
    index.add_collection(c_name, c_version)
    index.write_paraphrases(f"{c_name}.jsonl")
    index.db.close()
    del index
    extractor = CorpusPairExtractor(dataset)
    extractor.extract(f"{c_name}.jsonl", f"{c_name}_pairs.jsonl")


if __name__ == '__main__':
    fire.Fire(run_mining)
