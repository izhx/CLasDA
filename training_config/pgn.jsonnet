// Import template file.
local template = import "adapter-bert-base.libsonnet";

local data () = {
    reader: {
        type: "conll2003_ner_crowd",
    },
    path: {
        prefix: "data/conll03/",
        train: self.prefix + "answers.txt",
        dev: self.prefix + "dev.bio",
        test: self.prefix + "test.bio"
    },
};

local model () = {
    type: "pgn_crf_tagger",
    worker_num: 48,
    matched_embedder: {
        domain_num: $["worker_num"],
        domain_embedding_dim: 8,
        pgn_layers: std.parseInt(std.extVar("PGN_LAYERS")),
        share_param: false,
    },
    // pgn_initializer: {
    //     regexes: [
    //         ["t_w_down", {"type": "normal", "std": 1e-3}],
    //         ["t_w_up", {"type": "normal", "std": 1e-3}],
    //     ]
    // }
};

template(data(), model())
