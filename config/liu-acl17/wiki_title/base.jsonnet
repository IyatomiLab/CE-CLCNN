local cuda_device = std.parseInt(std.extVar("GPU"));

local seed = 19950815;
local num_epochs = 20;
local batch_size = 128;

local character_tokenizer = {};

local tng_dataset_reader = {
    type: "wiki_title",
    character_tokenizer: character_tokenizer,
    manual_distributed_sharding: true,
    manual_multiprocess_sharding: true,
};

local val_dataset_reader = {
    type: "wiki_title",
    character_tokenizer: character_tokenizer,
    manual_distributed_sharding: true,
    manual_multiprocess_sharding: true,
};

local tng_data_loader = {
    batch_size: batch_size,
    drop_last: true,
    shuffle: true,
};

local val_data_loader = {
    batch_size: batch_size,
};

local optimizer = {
    type: "adam",
};

local trainer = {
    optimizer: optimizer,
    validation_metric: "+acc1",
    num_epochs: num_epochs,
    cuda_device: cuda_device,
    callbacks: [
        {type: "track_epoch_callback"},
        {type: "tensorboard", should_log_learning_rate: true},
    ],
};

{
    random_seed: seed,
    numpy_seed: seed,
    pytorch_seed: seed,

    datasets_for_vocab_creation: ["train"],
    evaluate_on_test: true,

    dataset_reader: tng_dataset_reader,
    validation_dataset_reader: val_dataset_reader,

    data_loader: tng_data_loader,
    validation_data_loader: val_data_loader,

    trainer: trainer
}
