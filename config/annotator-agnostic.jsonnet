// Import template file.
local template = import "adapter-bert-base.libsonnet";

local data () = {
    reader: {
        type: "conll2003_ner_crowd",
    },
    path: {
        prefix: "data/conll03/",
        train: self.prefix + std.extVar("TRAIN_FILE") + ".txt",
        dev: self.prefix + "dev.bio",
        test: self.prefix + "test.bio"
    },
};

local model () = {
    type: "my_crf_tagger",
    matched_embedder: { external_param: false }
};

local trainer () = {
    num_epochs: 20,
    // grad_norm: 5.0,
    patience: 5,
    optimizer: {
        type: "huggingface_adamw",
        lr: 1e-3,
        weight_decay: 0.01,
        correct_bias: true,
        parameter_groups: [[[".*transformer_model.*"], {"lr": 1e-5}]]
    },
};

template(data(), model(), trainer())
