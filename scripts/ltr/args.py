import argparse


def _add_dataset_creation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--custom-datasets",
        nargs="+",
        type=str,
        required=True,
        help="One or more raw custom dataset names under data/custom_datasets/.",
    )

    parser.add_argument(
        "--formatted-dataset-name",
        type=str,
        default=None,
        help=(
            "Name used for saving the formatted dataset. "
            "If omitted and exactly one --custom-datasets value is supplied, "
            "that dataset name is used."
        ),
    )

    parser.add_argument(
        "--layer-type",
        type=str,
        default="clumsy",
        choices=["clumsy", "trad", "mix", "all"],
        help="The perturbation type to use.",
    )

    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--random-pairs",
        action="store_true",
        default=False,
        help=(
            "Instead of using all items from the same chain, construct random "
            "pairs. The resulting examples always contain exactly two texts."
        ),
    )

    parser.add_argument(
        "--reuse-limit",
        type=int,
        default=5,
        help=(
            "Maximum number of times a single text can appear when constructing "
            "random pairs."
        ),
    )

    parser.add_argument(
        "--heldout-ratio",
        type=float,
        default=0.3,
        help=(
            "Fraction of the full dataset reserved for dev+test. "
            "Default reproduces the old behavior: 70 train / 30 heldout."
        ),
    )

    parser.add_argument(
        "--test-ratio-within-heldout",
        type=float,
        default=0.5,
        help=(
            "Fraction of heldout used as test. Default 0.5 gives "
            "70 train / 15 dev / 15 test when heldout-ratio is 0.3."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing formatted dataset directory.",
    )


def _add_saved_dataset_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--formatted-dataset-name",
        type=str,
        default=None,
        help=(
            "Name of a previously-created formatted dataset. The dataset will "
            "be loaded from data/custom_datasets/{name}/formatted_datasets/."
        ),
    )

    group.add_argument(
        "--formatted-dataset-path",
        type=str,
        default=None,
        help="Explicit path to a saved Hugging Face DatasetDict.",
    )


def parse_ds_create_args():
    parser = argparse.ArgumentParser(
        description="Create and save a fixed train/dev/test dataset for LTR training."
    )

    _add_dataset_creation_args(parser)

    return parser.parse_args()


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate an LTR model using a preformatted dataset."
    )

    parser.add_argument("model_name", type=str)
    parser.add_argument("max_seq_len", type=int)

    _add_saved_dataset_args(parser)

    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--loss",
        type=str,
        default="logistic",
        choices=[
            "logistic",
            "pairwise_logistic",
            "hinge",
            "margin",
            "weighted_logistic",
            "logistic_weighted",
            "weighted-logistic",
        ],
        help="Pairwise ranking loss to use.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="items",
        choices=["pairs", "items"],
    )

    parser.add_argument("--length_diagnostics", action="store_true")
    parser.add_argument("--length_plot_num_bins", type=int, default=10)
    parser.add_argument("--length_plot_max_pairs", type=int, default=200000)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--fsdp_layer_cls", type=str, default=None)

    parser.add_argument(
        "--hpo_mode",
        action="store_true",
        help="If set, save post-training dev metrics for HPO selection.",
    )

    parser.add_argument(
        "--skip_final_test_eval",
        action="store_true",
        help="If set, do not evaluate the held-out test split after training.",
    )

    parser.add_argument(
        "--hpo_metric_prefix",
        type=str,
        default="hpo_dev",
        help="Metric prefix used when saving post-training dev metrics.",
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Skip training and only run final evaluation with the supplied model.",
    )

    return parser.parse_args()


# Backwards-compatible name for training script.
def parse_args():
    return parse_train_args()