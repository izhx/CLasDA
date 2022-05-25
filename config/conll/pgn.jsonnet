// Import template file.
local template = import "template.libsonnet";

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
    embedder_key: "bert",
    reduction: "mean",
    pgn: {
        worker_num: 48,
        worker_dim: 8,
        pgn_layers: std.parseInt(std.extVar("PGN_LAYERS")),
        share_param: false,
    },
    adapter: { external_param: true }
};

local trainer () = {
};

template(data(), model(), trainer())
