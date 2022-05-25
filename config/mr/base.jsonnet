// Import template file.
local template = import "template.libsonnet";

local data () = {
    reader: {
        type: "movie_reivew_crowd",
    },
    path: {
        prefix: "data/movie_review/",
        train: self.prefix + std.extVar("TRAIN_FILE") + ".txt",
        dev: self.prefix + "ratings_test.txt",
        test: null
    },
};

local model () = {
    type: "text_regressor",
    adapter: { external_param: false }
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
