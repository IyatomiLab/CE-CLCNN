local base = import "base.jsonnet";

local wildcard_ratio = 0.2;

base + {
    model+: {
        wildcard_ratio: wildcard_ratio,
    },
}
