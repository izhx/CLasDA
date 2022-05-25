local adapter_layers = std.parseInt(std.extVar("ADAPTER_LAYERS"));
local adapter_kwargs = {
    adapter_size: std.parseInt(std.extVar("ADAPTER_SIZE")),
    bias: true
};

local getattr (obj, name, default) = if std.objectHas(obj, name) then obj[name] else default;

function (data, model, trainer) {
    // [if std.objectHas(data, "exclude") then "exclude"]: data.exclude,
    dataset_reader : data.reader + { model_name: std.extVar("BACKBONE") },
    train_data_path: data.path.train,
    validation_data_path: data.path.dev,
    test_data_path: data.path.test,
    evaluate_on_test: if data.path.test == null then false else true,
    model: model + {
        adapter:: self.adapter, // hack for removing model.adapter
        backbone: {
            model_name: std.extVar("BACKBONE"),
            output_name: "pooler_output"
        } + if adapter_layers <= 0 then {
            type: "transformer"
        } else {
            type: "adapter_transformer",
            adapter: {
                layers: adapter_layers,
                kwargs: adapter_kwargs
            } + model.adapter
        }
        // dropout: getattr(model, "dropout", null),
        // initializer: {
        //     regexes: [
        //         ["tag_projection_layer.weight", {"type": "xavier_normal"}],
        //     ]
        // }
    },
    data_loader: {
        batch_size: std.parseInt(std.extVar("BATCH_SIZE")),
        shuffle: true
    },
    random_seed: std.parseInt(std.extVar("RANDOM_SEED")),
    numpy_seed: self.random_seed,
    pytorch_seed: self.random_seed,
    trainer: {
        cuda_device: 0,
        num_epochs: 25,
        // grad_norm: 5.0,
        patience: 5,
        validation_metric: "+accuracy",
        optimizer: {
            type: "adam",
            lr: 1e-3,
            parameter_groups: [[[".*transformer_model.*"], {"lr": 1e-5}]]
        },
        callbacks: [ { type: "distributed_test" } ],
        checkpointer: { keep_most_recent_by_count: 1 },
    } + trainer
}