local character_tokenizer = {};

local character_image_processor = {
    font_name: "dataset/fonts/NotoSansCJKjp-Regular.otf",
    font_size: 36,
};

local url = "https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/category_list_ja.txt";

{
    type: "wiki_title",
    character_tokenizer: character_tokenizer,
    character_image_processor: character_image_processor,
    category_list_path: url,
    manual_distributed_sharding: true,
    manual_multiprocess_sharding: true,
}
