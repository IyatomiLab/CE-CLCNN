local base = import "base.jsonnet";

local wildcard_ratio = 0.2;

local random_erasing = {
    p: 0.3,
    max_area_ratio: 0.4,
    min_area_ratio: 0.02,
    max_aspect_ratio: 2.0,
    min_aspect_ratio: 0.3,
};

local num_workers = 8;
local max_instances_in_memory = 512;

base + {
    //
    // with RE
    //
    dataset_reader+: {
        random_erasing: random_erasing,
    },
    data_loader+: {
        num_workers: num_workers,
        max_instances_in_memory: max_instances_in_memory,
    },
    validation_data_loader+: {
        num_workers: num_workers,
        max_instances_in_memory: max_instances_in_memory,
    },
    //
    // with WT
    //
    model+: {
        wildcard_ratio: wildcard_ratio,
    },
}
