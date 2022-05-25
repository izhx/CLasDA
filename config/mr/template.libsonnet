local adapter_layers = std.parseInt(std.extVar("ADAPTER_LAYERS"));
local adapter_kwargs = {
    adapter_size: std.parseInt(std.extVar("ADAPTER_SIZE")),
    bias: true
};

local getattr (obj, name, default) = if std.objectHas(obj, name) then obj[name] else default;

function (data, model, trainer) {
    // [if std.objectHas(data, "exclude") then "exclude"]: data.exclude,
    dataset_reader : data.reader + {
        tokenizer: {
            type: "pretrained_transformer",
            model_name: std.extVar("BACKBONE")
        },
        token_indexers: {
            funnel: $["dataset_reader"].tokenizer
        }
    },
    train_data_path: data.path.train,
    validation_data_path: data.path.dev,
    test_data_path: data.path.test,
    evaluate_on_test: if data.path.test == null then false else true,
    model: model + {
        adapter:: self.adapter, // hack for removing model.adapter
        text_field_embedder: {
            token_embedders: {
                funnel: {
                    model_name: std.extVar("BACKBONE"),
                    class_name: "FunnelBaseModel"
                } + if adapter_layers <= 0 then {
                    type: "transformer"
                } else {
                    type: "adapter_transformer",
                    adapter: {
                        layers: adapter_layers,
                        kwargs: adapter_kwargs
                    } + model.adapter,
                }
            }
        },
        seq2vec_encoder: { type: "cls_pooler", embedding_dim: 768},
        feedforward: {
            input_dim: 768,
            num_layers: 1,
            hidden_dims: 768,
            activations: "tanh",
            dropout: 0.1
        },
        // dropout: getattr(model, "dropout", null),
        // initializer: {
        //     regexes: [
        //         ["tag_projection_layer.weight", {"type": "xavier_normal"}],
        //     ]
        // }
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
        validation_metric: "-mean_absolute_error",
        optimizer: {
            type: "adam",
            lr: 1e-3,
            parameter_groups: [[[".*transformer_model.*"], {"lr": 1e-5}]]
        },
        // callbacks: [ { type: "distributed_test" } ],
        checkpointer: { keep_most_recent_by_count: 1 },
    } + trainer
}