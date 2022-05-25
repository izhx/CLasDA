// Import template file.
local template = import "template.libsonnet";

local data () = {
    reader: {
        type: "movie_reivew_crowd",
    },
    path: {
        prefix: "data/movie_review/",
        train: self.prefix + "answers.txt",
        dev: self.prefix + "ratings_test.txt",
        test: null
    },
};

local model () = {
    type: "pgn_text_regressor",
    embedder_key: "funnel",
    pgn: {
        worker_num: 136,
        worker_dim: 8,
        pgn_layers: std.parseInt(std.extVar("PGN_LAYERS")),
        share_param: false,
    },
    adapter: { external_param: true }
};

local trainer () = {
    num_epochs: 30,
    // grad_norm: 5.0,
    patience: 5
};

template(data(), model(), trainer())
