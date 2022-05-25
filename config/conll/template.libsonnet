local tokenizer_kwargs = {
    do_lower_case: false
};
local adapter_layers = std.parseInt(std.extVar("ADAPTER_LAYERS"));
local adapter_kwargs = {
    adapter_size: std.parseInt(std.extVar("ADAPTER_SIZE")),
    bias: true
};

local getattr (obj, name, default) = if std.objectHas(obj, name) then obj[name] else default;

function (data, model, trainer) {
    // [if std.objectHas(data, "exclude") then "exclude"]: data.exclude,
    dataset_reader : data.reader + {
        token_indexers: {
            bert: {
                type: "pretrained_transformer_mismatched",
                model_name:
                    if std.objectHas(model, "bert_model_name")
                    then model.bert_model_name
                    else std.extVar("BACKBONE"),
                tokenizer_kwargs: getattr(model, "tokenizer_kwargs", tokenizer_kwargs)
            }
        },
        label_namespace: "labels"
    },
    train_data_path: data.path.train,
    validation_data_path: data.path.dev,
    test_data_path: data.path.test,
    evaluate_on_test: if data.path.test == null then false else true,
    model: model + {
        adapter:: self.adapter, // hack for removing model.adapter
        text_field_embedder: {
            token_embedders: {
                bert: {
                    type: "transformer_mismatched",
                    // sub_token_mode: "first",
                    matched_embedder: $["dataset_reader"].token_indexers.bert + if adapter_layers <= 0 then {
                        type: "transformer",
                    } else {
                        type: "adapter_transformer",
                        adapter: {
                            layers: adapter_layers,
                            kwargs: adapter_kwargs
                        } + model.adapter,
                    }
                },
            }
        },
        encoder: {
            type: "lstm",
            input_size: 768,
            hidden_size: 384,
            bidirectional: true,
            num_layers: 1
        },
        label_namespace: $["dataset_reader"].label_namespace,
        // dropout: getattr(model, "dropout", null),
        initializer: {
            regexes: [
                ["tag_projection_layer.weight", {"type": "xavier_normal"}],
                ["encoder._module.weight_ih.*", {"type": "xavier_normal"}],
                ["encoder._module.weight_hh.*", {"type": "orthogonal"}]
            ]
        }
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: std.parseInt(std.extVar("BATCH_SIZE")),
            sorting_keys: ["tokens"]
        }
    },
    random_seed: std.parseInt(std.extVar("RANDOM_SEED")),
    numpy_seed: self.random_seed,
    pytorch_seed: self.random_seed,
    trainer: {
        cuda_device: 0,
        num_epochs: 25,
        // grad_norm: 5.0,
        patience: 5,
        validation_metric: "+f1-measure-overall",
        optimizer: {
            type: "adam",
            lr: 1e-3,
            // parameter_groups: [[[".*transformer_model.*"], {"lr": 1e-5}]]
        },
        callbacks: [ { type: "distributed_test" } ],
        checkpointer: { keep_most_recent_by_count: 1 },
    } + trainer,
    vocabulary: { type: "from_files", directory: data.path.prefix + "vocabulary" }
}