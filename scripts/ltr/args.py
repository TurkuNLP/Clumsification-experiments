import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str)
    parser.add_argument("max_seq_len", type=int)
    parser.add_argument("--custom-datasets", nargs='+', type=str)
    parser.add_argument("--output-dir", type=str)

    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

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
        help=(
            "Pairwise ranking loss to use. "
            "Default is 'logistic', which preserves previous behavior."
        ),
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

    return parser.parse_args()