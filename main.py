"""
The debug wrapper script.
"""

import argparse
import os
import shutil
import sys

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--seed', '-s', type=str, default='42', help='random seed.')
_ARG_PARSER.add_argument('--name', '-n', type=str, default='debug', help='save name.')
_ARG_PARSER.add_argument('--debug', '-d', default=False, action="store_true")
_ARG_PARSER.add_argument('--config', type=str, default='pgn', help='configuration file name.')
_ARG_PARSER.add_argument('--predict', '-p', default=False, action="store_true")

_ARG_PARSER.add_argument('--bert_name', type=str, default='/nas-alinlp/linzhang.zx/models/bert-base-cased/',
                         help='name or path to your bert model.')
_ARG_PARSER.add_argument('--train_file', type=str, default='answers', help='mv, answers, ground_truth')
_ARG_PARSER.add_argument('--adapter_size', type=str, default='64', help='adapter bottleneck size')
_ARG_PARSER.add_argument('--adapter_layers', type=str, default='12', help='adapter bert layers')

_ARG_PARSER.add_argument('--pgn_layers', type=str, default='12', help='pgn layers')

_ARGS = _ARG_PARSER.parse_args()

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda
os.environ['RANDOM_SEED'] = _ARGS.seed

os.environ['BERT_MODEL_NAME'] = _ARGS.bert_name
os.environ['TRAIN_FILE'] = _ARGS.train_file

os.environ['ADAPTER_SIZE'] = _ARGS.adapter_size
os.environ['ADAPTER_LAYERS'] = _ARGS.adapter_layers

os.environ['PGN_LAYERS'] = _ARGS.pgn_layers

if _ARGS:
    from allennlp.commands import main

serialization_dir = f"results/{_ARGS.name}-{_ARGS.seed}"

# Assemble the command into sys.argv
argv = ["allennlp"]  # command name, not used by main
if _ARGS.predict:
    argv += [
        "predict",
        serialization_dir + "/model.tar.gz",  # archive_file
        "data/INPUT_FILE",  # TODO: input_file
        "--output-file", serialization_dir + "/predict.json",
        "--silent",
        "--cuda-device", _ARGS.cuda,
        "--use-dataset-reader",
    ]
else:
    if _ARGS.debug:
        # Training will fail if the serialization directory already
        # has stuff in it. If you are running the same training loop
        # over and over again for debugging purposes, it will.
        # Hence we wipe it out in advance.
        # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
        shutil.rmtree(serialization_dir, ignore_errors=True)
    argv += [
        "train",
        f"config/{_ARGS.config}.jsonnet",
        "-s", serialization_dir,
        # "--include-package", "clasda",
    ]

if not _ARGS.debug:
    argv.append("--file-friendly-logging")

print(" ".join(argv))
sys.argv = argv
main()
