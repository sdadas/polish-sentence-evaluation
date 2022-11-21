### Evaluation of Sentence Representations in Polish
This repository contains experiments related to dense representations of sentences in Polish. It includes code for evaluating different sentence representation methods such as aggregated word embeddings or neural sentence encoders, both multilingual and language-specific. This source code has been used in the following publications:


#### [[1]](https://aclanthology.org/2020.lrec-1.207/) Evaluation of Sentence Representations in Polish 

The paper contains evaluation of eight sentence representation methods (Word2Vec, GloVe, FastText, ELMo, Flair, BERT, LASER, USE) on five polish linguistic tasks.
Dataset for these tasks are distributed with the repository and two of them are released specifically for this evaluation:
the [SICK (Sentences Involving Compositional Knowledge)](https://github.com/text-machine-lab/MUTT/tree/master/data/sick) corpus translated to Polish and 8TAGS classification dataset. Pre-trained models used in this study are available for download in separate repository: [Polish NLP Resources](https://github.com/sdadas/polish-nlp-resources).

<details>
  <summary>BibTeX</summary>
  
  ```
  @inproceedings{dadas-etal-2020-evaluation,
    title = "Evaluation of Sentence Representations in {P}olish",
    author = "Dadas, Slawomir  and Pere{\l}kiewicz, Micha{\l} and Po{\'s}wiata, Rafa{\l}",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.207",
    pages = "1674--1680",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
  ```
</details>

#### [[2]](https://arxiv.org/abs/2207.12759) Training Effective Neural Sentence Encoders from Automatically Mined Paraphrases

In this publication, we show a simple method for training effective language-specific sentence encoders without manually labeled data. Our approach is to automatically construct a dataset of paraphrase pairs from sentence-aligned bilingual text corpora. We then use the collected data to fine-tune a Transformer language model with an additional recurrent pooling layer.

<details>
  <summary>BibTeX</summary>
  
```
@inproceedings{9945218,
  author={Dadas, S{\l}awomir},
  booktitle={2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)}, 
  title={Training Effective Neural Sentence Encoders from Automatically Mined Paraphrases}, 
  year={2022},
  volume={},
  number={},
  pages={371-378},
  doi={10.1109/SMC53654.2022.9945218}
}
```
</details>

### Updates:

- **20.01.2022** - [New code example](https://github.com/sdadas/polish-sentence-evaluation/tree/master/examples/paraphrase_mining) added: training sentence encoders on paraphrase pairs mined from OPUS parallel corpus.
- **23.10.2020** - Added pre-trained multilingual models from the [Sentence-Transformers](https://www.sbert.net/) library
- **02.09.2020** - Added [LaBSE](https://tfhub.dev/google/LaBSE/1) multilingual sentence encoder
- **09.05.2020** - Added new [Polish RoBERTa](https://github.com/sdadas/polish-roberta) models
- **03.03.2020** - Added [XLM-RoBERTa (base)](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) model
- **02.02.2020** - Added detailed results of static word embedding models with dimensionalities from 300 to 800
- **01.02.2020** - Added [Polish RoBERTa](https://github.com/sdadas/polish-nlp-resources#roberta) model and multilingual [XLM-RoBERTa (large)](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) model

### Evaluation results:
<table>
  <thead>
    <th><strong>#</strong></th>
    <th><strong>Method</strong></th>
    <th><strong>Language</strong></th>
    <th><strong>WCCRS<br/>Hotels</strong></th>
    <th><strong>WCCRS<br/>Medicine</strong></th>
    <th><strong>SICK‑E</strong></th>
    <th><strong>SICK‑R</strong></th>
    <th><strong>8TAGS</strong></th>
  </thead>
  <tr>
    <td colspan="8"><strong>Word embeddings</strong></td>                                                                     
  </tr>
  <tr><td>1</td><td>Random</td><td>n/a</td><td>65.83</td><td>60.64</td><td>72.77</td><td>0.628</td><td>31.95</td></tr>
    <tr><td>2.a</td><td>Word2Vec (300d)</td><td>Polish</td><td>78.19</td><td>73.23</td><td>75.42</td><td>0.746</td><td>70.27</td></tr>
  <tr><td>2.b</td><td>Word2Vec (500d)</td><td>Polish</td><td>81.72</td><td>73.98</td><td>76.25</td><td>0.764</td><td>70.56</td></tr>
  <tr><td>2.c</td><td>Word2Vec (800d)</td><td>Polish</td><td><strong>82.24</strong></td><td>73.88</td><td>75.60</td><td>0.772</td><td><strong>70.79</strong></td></tr>
    <tr><td>3.a</td><td>GloVe (300d)</td><td>Polish</td><td>80.05</td><td>72.54</td><td>73.81</td><td>0.756</td><td>69.78</td></tr>
  <tr><td>3.b</td><td>GloVe (500d)</td><td>Polish</td><td>80.76</td><td>72.54</td><td>75.09</td><td>0.761</td><td>70.27</td></tr>
  <tr><td>3.c</td><td>GloVe (800d)</td><td>Polish</td><td>81.79</td><td><strong>74.32</strong></td><td>76.48</td><td><strong>0.779</strong></td><td>70.63</td></tr>
    <tr><td>4.a</td><td>FastText (300d)</td><td>Polish</td><td>80.31</td><td>72.64</td><td>75.19</td><td>0.729</td><td>69.24</td></tr>
  <tr><td>4.b</td><td>FastText (500d)</td><td>Polish</td><td>80.31</td><td>73.88</td><td>76.66</td><td>0.755</td><td>70.22</td></tr>
  <tr><td>4.c</td><td>FastText (800d)</td><td>Polish</td><td>80.95</td><td>72.94</td><td><strong>77.09</strong></td><td>0.768</td><td>69.95</td></tr>
  <tr>
    <td colspan="8"><strong>Language models</strong></td>
  </tr>
  <tr><td>5.a</td><td>ELMo (all)</td><td>Polish</td><td>85.52</td><td>78.42</td><td>77.15</td><td>0.789</td><td>71.41</td></tr>
    <tr><td>5.b</td><td>ELMo (top)</td><td>Polish</td><td>83.20</td><td>78.17</td><td>74.05</td><td>0.756</td><td>71.41</td></tr>
    <tr><td>6</td><td>Flair</td><td>Polish</td><td>80.82</td><td>75.46</td><td><strong>78.43</strong></td><td>0.743</td><td>65.62</td></tr>
  <tr><td>7.a</td><td>RoBERTa-base (all)</td><td>Polish</td><td>85.78</td><td>78.96</td><td>78.82</td><td>0.799</td><td>70.27</td></tr> 
  <tr><td>7.b</td><td>RoBERTa-base (top)</td><td>Polish</td><td>84.62</td><td>79.36</td><td>76.09</td><td>0.750</td><td>70.33</td></tr>
  <tr><td>7.c</td><td>RoBERTa-large (all)</td><td>Polish</td><td><strong>89.12</strong></td><td><strong>84.74</strong></td><td>78.13</td><td><strong>0.820</strong></td><td>75.75</td></tr> 
  <tr><td>7.d</td><td>RoBERTa-large (top)</td><td>Polish</td><td>88.93</td><td>83.11</td><td>75.56</td><td>0.767</td><td><strong>76.67</strong></td></tr>
  <tr><td>8.a</td><td>XLM-RoBERTa-base (all)</td><td>Multilingual</td><td>85.52</td><td>78.81</td><td>75.25</td><td>0.734</td><td>68.78</td></tr> 
  <tr><td>8.b</td><td>XLM-RoBERTa-base (top)</td><td>Multilingual</td><td>82.37</td><td>75.26</td><td>64.47</td><td>0.579</td><td>69.81</td></tr> 
  <tr><td>8.c</td><td>XLM-RoBERTa-large (all)</td><td>Multilingual</td><td>87.39</td><td>83.60</td><td>74.34</td><td>0.764</td><td>73.33</td></tr> 
  <tr><td>8.d</td><td>XLM-RoBERTa-large (top)</td><td>Multilingual</td><td>85.07</td><td>78.91</td><td>61.50</td><td>0.568</td><td>73.35</td></tr> 
    <tr><td>9</td><td>BERT</td><td>Multilingual</td><td>76.83</td><td>72.54</td><td>73.83</td><td>0.698</td><td>65.05</td></tr>
  <tr>
    <td colspan="8"><strong>Sentence encoders</strong></td>
  </tr>
  <tr><td>10</td><td>LASER</td><td>Multilingual</td><td>81.21</td><td>78.17</td><td><strong>82.21</strong></td><td>0.825</td><td>64.91</td></tr>
    <tr><td>11</td><td>USE</td><td>Multilingual</td><td>79.47</td><td>73.78</td><td>82.14</td><td>0.833</td><td>69.92</td></tr>
  <tr><td>12</td><td>LaBSE</td><td>Multilingual</td><td><strong>85.52</strong></td><td><strong>80.89</strong></td><td>81.57</td><td>0.825</td><td><strong>72.35</strong></td></tr>
  <tr><td>13a</td><td>Sentence-Transformers<br/><sup>(distiluse-base-multilingual-cased-v2)</sup></td><td>Multilingual</td><td>79.99</td><td>75.80</td><td>78.90</td><td>0.807</td><td>70.86</td></tr>
  <tr><td>13b</td><td>Sentence-Transformers<br/><sup>(xlm-r-distilroberta-base-paraphrase-v1)</sup></td><td>Multilingual</td><td>82.63</td><td>80.84</td><td>81.35</td><td><strong>0.839</strong></td><td>70.61</td></tr>
  <tr><td>13c</td><td>Sentence-Transformers<br/><sup>(xlm-r-bert-base-nli-stsb-mean-tokens)</sup></td><td>Multilingual</td><td>81.02</td><td>79.95</td><td>79.09</td><td>0.820</td><td>69.12</td></tr>
  <tr><td>13d</td><td>Sentence-Transformers<br/><sup>(distilbert-multilingual-nli-stsb-quora-ranking)</sup></td><td>Multilingual</td><td>80.05</td><td>74.64</td><td>79.41</td><td>0.817</td><td>69.28</td></tr>
</table>

Table: Evaluation of sentence representations on four classification tasks and one semantic relatedness task (SICK-R). For classification, we report accuracy of each model. For semantic relatedness, Pearson correlation between true and predicted relatedness scores is reported.

### Evaluated methods:

1. Randomly initialized word embeddings
2. Word2Vec ([Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)) model pre-trained by us. The number in parentheses indicates the dimensionality of the embeddings.
3. GloVe ([Glove: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162.pdf)) model pre-trained by us. The number in parentheses indicates the dimensionality of the embeddings. [[Download]](https://github.com/sdadas/polish-nlp-resources#glove)
4. FastText ([Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)) model pre-trained by us. The number in parentheses indicates the dimensionality of the embeddings.
5. ELMo language model described in [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) paper, pre-trained by us for Polish. In the `all` variant, we construct the word representation by concatenating all hidden states of the LM. In the `top` variant, only the top LM layer is used as word representation. [[Download]](https://github.com/sdadas/polish-nlp-resources#elmo)
6. Flair language model described in [Contextual String Embeddings for Sequence Labeling](https://www.aclweb.org/anthology/C18-1139.pdf). We concatenate the outputs of the original `pl-forward` and `pl-backward` pre-trained language models available in the [Flair framework](https://github.com/flairNLP/flair).
7. RoBERTa language model described in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692), pre-trained by us for Polish. [[Download]](https://github.com/sdadas/polish-roberta)
8. XLM-RoBERTa is a large, multilingual language model trained by Facebook on 2.5 TB of text extracted from CommonCrawl. We evaluate two pre-trained architectures: base and large model. More information in their paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf). [[Download]](https://github.com/pytorch/fairseq/tree/master/examples/xlmr)
9. Original BERT language model by Google described in [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf). We use the `bert-base-multilingual-cased` version. [[Download]](https://github.com/google-research/bert/blob/master/multilingual.md)
10. Multilingual sentence encoder by Facebook, presented in [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/pdf/1812.10464.pdf). [[Download]](https://github.com/facebookresearch/LASER)
11. Multilingual sentence encoder by Google, presented in [Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/pdf/1907.04307.pdf).
12. [The language-agnostic BERT sentence embedding (LaBSE)](https://arxiv.org/pdf/2007.01852.pdf).
13. Pre-trained models from the [Sentence-Transformers](https://www.sbert.net/) library.

![results](results.png)

Figure: Evaluation of aggregation techniques for word embedding models with different dimensionalities. Baseline models use simple averaging, SIF is a method proposed by Arora et al. (2017), Max Pooling is a concatenation of arithmetic mean and max pooled vector from word embeddings.

### Usage

`evaluate_all.py` is used for evaluation of all available models. \
Run `evaluate.py [model_name] [model_params]` to evaluate single model. For example, `evaluate.py word2vec` runs evaluation on `word2vec_100_3_polish.bin` model.
Please note that in case of static embeddings and ELMo, you need to manually download the model from [Polish NLP Resources](https://github.com/sdadas/polish-nlp-resources) and place it in the `resources` directory.

### Acknowledgements
This evaluation is based on [SentEval](https://github.com/facebookresearch/SentEval) modified by us to support models, tasks and preprocessing for Polish language.
We'd like to thank authors of SentEval toolkit for making their code available. 

Two tasks in this study are based on [Wroclaw Corpus of Consumer Reviews](https://clarin-pl.eu/dspace/handle/11321/700).  We would like to thank the authors for making this data collection available.

