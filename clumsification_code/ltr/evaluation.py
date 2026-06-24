import math
import random
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset

from .diagnostics import (
    write_relative_length_winrate_plot,
    write_winrate_history_plot,
)
from .utils import logger


@torch.no_grad()
def evaluate_win_rate_distributed(
    model: nn.Module,
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batch_size: int = 4,
    collect_length_diagnostics: bool = False,
    length_plot_num_bins: int = 10,
    length_plot_max_pairs: int = 200000,
    length_diag_output_dir: Optional[str] = None,
    length_diag_step: Optional[int] = None,
    length_diag_epoch: Optional[float] = None,
    length_diag_seed: int = 0,
) -> Dict[str, float]:
    """
    Distributed evaluation.

    Main metric:
        win_rate over all unequal-label pairs.

    Tie-aware scoring:
        strict correct -> 1.0
        strict incorrect -> 0.0
        exact score tie -> 0.5
    """
    model.eval()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = next(model.parameters()).device
    dataset_len = len(dataset)

    if dataset_len == 0:
        return {
            "win_rate": 0.0,
            "correct_points": 0.0,
            "strict_correct_pairs": 0,
            "score_tie_rate": 0.0,
            "total_pairs": 0,
        }

    if collect_length_diagnostics:
        local_len_counts = torch.zeros(8, dtype=torch.float64, device=device)

        local_plot_pairs = []
        seen_plot_pairs = 0

        rng = random.Random(
            int(length_diag_seed)
            + 1_000_003 * int(rank)
            + 9_176 * int(length_diag_step or 0)
        )

        if length_plot_max_pairs < 0:
            local_plot_cap = None
        elif length_plot_max_pairs == 0:
            local_plot_cap = 0
        else:
            local_plot_cap = max(
                1,
                math.ceil(length_plot_max_pairs / max(world_size, 1)),
            )

        def maybe_add_plot_pair(row: tuple):
            nonlocal seen_plot_pairs, local_plot_pairs

            if local_plot_cap == 0:
                return

            seen_plot_pairs += 1

            if local_plot_cap is None:
                local_plot_pairs.append(row)
                return

            if len(local_plot_pairs) < local_plot_cap:
                local_plot_pairs.append(row)
            else:
                replace_idx = rng.randrange(seen_plot_pairs)
                if replace_idx < local_plot_cap:
                    local_plot_pairs[replace_idx] = row

    local_indices = list(range(rank, dataset_len, world_size))
    local_len = len(local_indices)

    local_len_tensor = torch.tensor([local_len], dtype=torch.long, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_len_tensor, op=dist.ReduceOp.MAX)

    max_local_len = int(local_len_tensor.item())

    padded_indices = local_indices + [-1] * (max_local_len - local_len)

    local_correct_points = 0.0
    local_total = 0
    local_score_tie_pairs = 0
    local_strict_correct_pairs = 0

    model_param_dtype = next(
        (
            p.dtype
            for p in model.parameters()
            if p.is_floating_point()
        ),
        torch.float32,
    )

    use_cuda_amp = (
        torch.cuda.is_available()
        and device.type == "cuda"
        and model_param_dtype in {torch.bfloat16, torch.float16}
    )

    amp_dtype = model_param_dtype

    if dist.is_available() and dist.is_initialized():
        logger.info(
            f"[rank {rank}] eval dataset_len={dataset_len}, "
            f"local_len={local_len}, max_local_len={max_local_len}, "
            f"batch_size={batch_size}, "
            f"num_eval_batches={math.ceil(max_local_len / batch_size)}"
        )

    for start in range(0, max_local_len, batch_size):
        batch_indices = padded_indices[start:start + batch_size]

        real_mask = [idx >= 0 for idx in batch_indices]
        fetch_indices = [idx if idx >= 0 else 0 for idx in batch_indices]

        rows = dataset[fetch_indices]

        texts_list = rows["texts"]
        labels_list = rows["labels"]

        group_sizes = [len(x) for x in texts_list]
        flat_texts = [t for group in texts_list for t in group]

        tok = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)

        gs = torch.tensor(group_sizes, dtype=torch.long, device=device)

        if use_cuda_amp:
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            ):
                out = model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    group_sizes=gs,
                    labels=None,
                )
        else:
            out = model(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                group_sizes=gs,
                labels=None,
            )

        scores = out["scores"]

        for b, labels in enumerate(labels_list):
            if not real_mask[b]:
                continue

            texts = texts_list[b]
            n = len(labels)

            if n < 2:
                continue

            s = scores[b, :n]
            l = torch.tensor(labels, device=device)

            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)

            li = l[idx_i]
            lj = l[idx_j]
            si = s[idx_i]
            sj = s[idx_j]

            valid = li != lj

            if not valid.any():
                continue

            pred_i_better = si > sj
            true_i_better = li < lj

            score_tie = torch.isclose(
                si.float(),
                sj.float(),
                rtol=0.0,
                atol=1e-6,
            )

            strict_pair_correct = pred_i_better == true_i_better

            pair_points = strict_pair_correct.to(torch.float32)
            pair_points = torch.where(
                score_tie,
                torch.full_like(pair_points, 0.5),
                pair_points,
            )

            valid_pair_points = pair_points[valid]

            local_correct_points += float(valid_pair_points.sum().item())
            local_total += int(valid.sum().item())

            local_score_tie_pairs += int(score_tie[valid].sum().item())
            local_strict_correct_pairs += int(
                ((~score_tie) & strict_pair_correct & valid).sum().item()
            )

            if collect_length_diagnostics:
                text_lengths = torch.tensor(
                    [len(x) for x in texts],
                    dtype=torch.long,
                    device=device,
                )

                len_i = text_lengths[idx_i]
                len_j = text_lengths[idx_j]

                valid_positions = torch.nonzero(valid, as_tuple=False).flatten()

                unequal_length = valid & (len_i != len_j)
                unequal_length_non_tie_score = unequal_length & (~score_tie)

                pred_prefers_shorter = (
                    (pred_i_better & (len_i < len_j))
                    | ((~pred_i_better) & (len_j < len_i))
                )

                true_shorter_better = (
                    ((li < lj) & (len_i < len_j))
                    | ((lj < li) & (len_j < len_i))
                )

                shorter_better_mask = unequal_length & true_shorter_better
                longer_better_mask = unequal_length & (~true_shorter_better)

                local_len_counts[0] += float(
                    pred_prefers_shorter[unequal_length_non_tie_score].sum().item()
                )
                local_len_counts[1] += float(
                    unequal_length_non_tie_score.sum().item()
                )

                local_len_counts[2] += float(
                    true_shorter_better[unequal_length].sum().item()
                )
                local_len_counts[3] += float(unequal_length.sum().item())

                local_len_counts[4] += float(
                    pair_points[shorter_better_mask].sum().item()
                )
                local_len_counts[5] += float(shorter_better_mask.sum().item())

                local_len_counts[6] += float(
                    pair_points[longer_better_mask].sum().item()
                )
                local_len_counts[7] += float(longer_better_mask.sum().item())

                if local_plot_cap != 0:
                    for pos_t in valid_positions:
                        pos = int(pos_t.item())

                        len_i_val = int(len_i[pos].item())
                        len_j_val = int(len_j[pos].item())

                        denom = max(len_i_val, len_j_val, 1)
                        rel_len_diff = abs(len_i_val - len_j_val) / float(denom)

                        point_value = float(pair_points[pos].item())

                        maybe_add_plot_pair(
                            (
                                float(rel_len_diff),
                                point_value,
                            )
                        )

    counts = torch.tensor(
        [
            local_correct_points,
            float(local_total),
            float(local_score_tie_pairs),
            float(local_strict_correct_pairs),
        ],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    correct_points = float(counts[0].item())
    total = int(round(float(counts[1].item())))
    score_tie_pairs = int(round(float(counts[2].item())))
    strict_correct_pairs = int(round(float(counts[3].item())))

    win_rate = correct_points / total if total > 0 else 0.0

    metrics = {
        "win_rate": win_rate,
        "correct_points": correct_points,
        "strict_correct_pairs": strict_correct_pairs,
        "score_tie_rate": score_tie_pairs / total if total > 0 else 0.0,
        "total_pairs": total,
    }

    if collect_length_diagnostics:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_len_counts, op=dist.ReduceOp.SUM)

        len_counts = local_len_counts.detach().cpu().tolist()

        model_prefers_shorter = float(len_counts[0])
        unequal_len_non_tie = float(len_counts[1])

        true_shorter_better_count = float(len_counts[2])
        unequal_len = float(len_counts[3])

        correct_points_when_shorter_better = float(len_counts[4])
        total_shorter_better = float(len_counts[5])

        correct_points_when_longer_better = float(len_counts[6])
        total_longer_better = float(len_counts[7])

        metrics["model_prefers_shorter_rate"] = (
            model_prefers_shorter / unequal_len_non_tie
            if unequal_len_non_tie > 0
            else 0.0
        )

        metrics["true_shorter_better_rate"] = (
            true_shorter_better_count / unequal_len
            if unequal_len > 0
            else 0.0
        )

        metrics["win_rate_when_shorter_is_better"] = (
            correct_points_when_shorter_better / total_shorter_better
            if total_shorter_better > 0
            else 0.0
        )

        metrics["win_rate_when_longer_is_better"] = (
            correct_points_when_longer_better / total_longer_better
            if total_longer_better > 0
            else 0.0
        )

        if length_plot_max_pairs != 0:
            if dist.is_available() and dist.is_initialized():
                gathered_plot_pairs = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_plot_pairs, local_plot_pairs)
            else:
                gathered_plot_pairs = [local_plot_pairs]

            if rank == 0:
                all_plot_pairs = []

                for rows_for_rank in gathered_plot_pairs:
                    if rows_for_rank:
                        all_plot_pairs.extend(rows_for_rank)

                if (
                    length_plot_max_pairs > 0
                    and len(all_plot_pairs) > length_plot_max_pairs
                ):
                    rng_global = random.Random(
                        int(length_diag_seed)
                        + 53_111 * int(length_diag_step or 0)
                    )
                    all_plot_pairs = rng_global.sample(
                        all_plot_pairs,
                        length_plot_max_pairs,
                    )

                write_winrate_history_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    metrics=metrics,
                )

                write_relative_length_winrate_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    diagnostic_pairs=all_plot_pairs,
                    num_bins=length_plot_num_bins,
                )

                logger.info(
                    f"Wrote length diagnostic plots with "
                    f"{len(all_plot_pairs)} diagnostic pairs."
                )
        else:
            if rank == 0:
                write_winrate_history_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    metrics=metrics,
                )

    return metrics


def baseline_winrates(dataset: Dataset) -> Dict[str, float]:
    """
    Computes simple baselines over valid unequal-label pairs.

    Lower label means better item.
    """
    total = 0
    first_correct = 0
    shorter_correct = 0.0

    for texts, labels in zip(dataset["texts"], dataset["labels"]):
        n = len(labels)

        for i in range(n - 1):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue

                total += 1

                i_better = labels[i] < labels[j]

                if i_better:
                    first_correct += 1

                len_i = len(texts[i])
                len_j = len(texts[j])

                if len_i < len_j:
                    shorter_correct += float(i_better)
                elif len_j < len_i:
                    shorter_correct += float(not i_better)
                else:
                    shorter_correct += 0.5

    if total == 0:
        return {
            "random_baseline": 0.0,
            "first_item_baseline": 0.0,
            "shorter_text_baseline": 0.0,
            "total_valid_pairs": 0,
        }

    return {
        "random_baseline": 0.5,
        "first_item_baseline": first_correct / total,
        "shorter_text_baseline": shorter_correct / total,
        "total_valid_pairs": total,
    }