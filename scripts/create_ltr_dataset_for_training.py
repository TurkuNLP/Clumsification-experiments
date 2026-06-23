import os

import dataset_functions as d_f
from ltr.args import parse_ds_create_args
from ltr.utils import configure_logging, logger


def resolve_formatted_dataset_name(args) -> str:
    if args.formatted_dataset_name is not None:
        return args.formatted_dataset_name

    if len(args.custom_datasets) == 1:
        return args.custom_datasets[0]

    raise ValueError(
        "When using multiple --custom-datasets, you must provide "
        "--formatted-dataset-name because the save location would otherwise "
        "be ambiguous."
    )


def main():
    configure_logging()

    args = parse_ds_create_args()

    formatted_dataset_name = resolve_formatted_dataset_name(args)
    output_path = d_f.default_formatted_dataset_path(formatted_dataset_name)

    logger.info("Creating formatted dataset")
    logger.info(f"Raw dataset names: {args.custom_datasets}")
    logger.info(f"Formatted dataset name: {formatted_dataset_name}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"layer_type={args.layer_type}")
    logger.info(f"max_layers={args.max_layers}")
    logger.info(f"random_pairs={args.random_pairs}")
    logger.info(f"reuse_limit={args.reuse_limit}")
    logger.info(f"seed={args.seed}")
    logger.info(f"downsample_size={args.downsample_size}")

    dataset_dict = d_f.create_formatted_dataset_dict(
        dataset_names=args.custom_datasets,
        max_layers=args.max_layers,
        layer_type=args.layer_type,
        seed=args.seed,
        random_pairs=args.random_pairs,
        reuse_limit=args.reuse_limit,
        downsample_size=args.downsample_size,
        heldout_ratio=args.heldout_ratio,
        test_ratio_within_heldout=args.test_ratio_within_heldout,
    )

    metadata = {
        "custom_datasets": args.custom_datasets,
        "formatted_dataset_name": formatted_dataset_name,
        "layer_type": args.layer_type,
        "max_layers": args.max_layers,
        "random_pairs": args.random_pairs,
        "reuse_limit": args.reuse_limit,
        "seed": args.seed,
        "downsample_size": args.downsample_size,
        "heldout_ratio": args.heldout_ratio,
        "test_ratio_within_heldout": args.test_ratio_within_heldout,
        "num_train": len(dataset_dict["train"]),
        "num_dev": len(dataset_dict["dev"]),
        "num_test": len(dataset_dict["test"]),
    }

    d_f.save_formatted_dataset_dict(
        dataset_dict=dataset_dict,
        output_path=output_path,
        metadata=metadata,
        overwrite=args.overwrite,
    )

    logger.info("Saved formatted dataset")
    logger.info(f"Path: {output_path}")
    logger.info(f"train={len(dataset_dict['train'])}")
    logger.info(f"dev={len(dataset_dict['dev'])}")
    logger.info(f"test={len(dataset_dict['test'])}")


if __name__ == "__main__":
    main()