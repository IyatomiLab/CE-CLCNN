local base = import "../base.jsonnet";

local category_list_path = "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_ko.txt";

local tng_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_train.txt";
local val_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_val.txt";
local tst_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_test.txt";

local char_embed_dim = 128;
# ref. https://github.com/frederick0329/Learning-Character-Level/blob/57e90e1fa7f026025f4af8b120d0b714326ce5b7/code/doall.lua#L25
local rnn_hidden_size = 400;

local lookup_encoder = {
    type: "basic",
    token_embedders: {
        chars: {
            type: "embedding",
            embedding_dim: char_embed_dim,
            vocab_namespace: "chars",
            trainable: true,
        },
    },
};

local model = {
    type: "liu_acl17_lookup",
    lookup_encoder: lookup_encoder,
    rnn_encoder: {
        input_size: char_embed_dim,
        hidden_size: rnn_hidden_size, 
    },
};

base + {
    dataset_reader+: {
        category_list_path: category_list_path,
    },
    validation_dataset_reader+: {
        category_list_path: category_list_path,
    },
    
    train_data_path: tng_data_path,
    validation_data_path: val_data_path,
    test_data_path: tst_data_path,

    model: model,
}
