// Import template file.
local template = import "template.libsonnet";

local data () = {
    reader: {
        type: "label_me_crowd",
    },
    path: {
        prefix: "data/label_me/",
        train: self.prefix + std.extVar("TRAIN_FILE") + ".txt",
        dev: self.prefix + "labels_valid.txt",
        test: self.prefix + "labels_test.txt"
    },
};

local model () = {
    type: "image_classifier",
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
