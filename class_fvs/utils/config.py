
CONFIGS = {
    "resnet50": {
        "layer_weights": [1, 1, 1, 1, 1],
        "tv_l2": 0.0001,
        "l2": 0.0001,
        "lambda_graph": 0.5,
        "graph_K": 32
    },
    "vit_l_16": {
        "layer_weights": [1, 1, 1, 1, 1],
        "tv_l2": 0.003,
        "l2": 0.003,
        "lambda_graph": 1,
        "graph_K": 32
    },
    "vit_l_32": {
        "layer_weights": [1, 1, 1, 1, 0.1],
        "tv_l2": 0.0003,
        "l2": 0.0003,
        "lambda_graph": 1,
        "graph_K": 10
    },
    "convnext_base": {
        "layer_weights": [10, 1, 1, 1, 1],
        "tv_l2": 0,
        "l2": 0,
        "lambda_graph": 0.25,
        "graph_K": 10
    }
}

SPLIT_SEED = 42
REFERENCE_SPLIT = "selection"   # "selection" or "heldout"

def get_config(args):
    defaults = CONFIGS[args.arch_name]

    if getattr(args, "layer_weights", None) is None:
        args.layer_weights = defaults["layer_weights"]

    if getattr(args, "lambda_graph", None) is None:
        args.lambda_graph = defaults["lambda_graph"]

    if getattr(args, "graph_K", None) is None:
        args.graph_K = defaults["graph_K"]

    if getattr(args, "tv_l2", None) is None:
        args.tv_l2 = defaults["tv_l2"]

    if getattr(args, "l2", None) is None:
        args.l2 = defaults["l2"]
        
    # Fixed reference-data partition
    args.split_seed = SPLIT_SEED
    args.reference_split = REFERENCE_SPLIT

    # Optimization seed = run/trial ID
    args.seed = args.run_id


    args.setup_folder = getattr(args, 'setup_folder', "class_neurons")

    args.folder_name = getattr(
        args,
        'folder_name',
        f"{args.setup_folder}/{args.arch_name}/cls{args.target}/t{args.run_id}"
    )

    return args