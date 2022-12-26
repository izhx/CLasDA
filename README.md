# Crowdsourcing Learning as Domain Adaptation.

Code for ACL-IJCNLP 2021 paper: "[Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition](https://arxiv.org/abs/2105.14980)"

This is the partially re-writted code based-on [AllenNLP](https://github.com/allenai/allennlp). I've reproduced the unsupervised setting results.

For the original experiment code and data, please refer to [crowd-NER](https://github.com/izhx/crowd-NER).


## Usage

- GOLD: `python main.py --name=gold --train_file=ground_truth --config=annotator-agnostic`
- ALL: `python main.py --name=all --train_file=answers --config=annotator-agnostic`
- MV: `python main.py --name=mv --train_file=mv --config=annotator-agnostic`
- Our model: `python main.py --name=pgn --train_file=answers --config=pgn`

## Citation

```
@inproceedings{zhang-etal-2021-crowdsourcing,
    title = "Crowdsourcing Learning as Domain Adaptation: {A} Case Study on Named Entity Recognition",
    author = "Zhang, Xin  and
      Xu, Guangwei  and
      Sun, Yueheng  and
      Zhang, Meishan  and
      Xie, Pengjun",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.432",
    doi = "10.18653/v1/2021.acl-long.432",
    pages = "5558--5570",
}
```
