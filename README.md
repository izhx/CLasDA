# Crowdsourcing Learning as Domain Adaptation.

Code for ACL-IJCNLP 2021 paper: "[Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition](https://arxiv.org/abs/2105.14980)"

Coming soon!

I'm trying to rewrite the code with allennlp.

For the original experiment code, please refer to [crowd-NER](https://github.com/izhx/crowd-NER).


## Usage

- GOLD: `python main.py --name=gold --train_file=ground_truth --config=annotator-agnostic`
- ALL: `python main.py --name=all --train_file=answers --config=annotator-agnostic`
- MV: `python main.py --name=mv --train_file=mv --config=annotator-agnostic`
- Our model: `python main.py --name=pgn --train_file=answers --config=pgn`
