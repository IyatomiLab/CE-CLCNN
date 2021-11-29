local base = import "../base.jsonnet";

local font_name = "dataset/fonts/NotoSansCJKjp-Regular.otf";
local category_list_ja_path = "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_ja.txt";

local tng_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_train.txt";
local val_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_val.txt";
local tst_data_path = "https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ko_test.txt";

base + {
    dataset_reader+: {
        category_list_path: category_list_ja_path,
        character_image_processor+: {
            font_name: font_name,
        },
    },
    validation_dataset_reader+: {
        category_list_path: category_list_ja_path,
        character_image_processor+: {
            font_name: font_name,
        },
    },
    
    train_data_path: tng_data_path,
    validation_data_path: val_data_path,
    test_data_path: tst_data_path,
}
