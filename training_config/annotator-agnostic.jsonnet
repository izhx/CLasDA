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
    type: "my_crf_tagger"
};

template(data(), model())
