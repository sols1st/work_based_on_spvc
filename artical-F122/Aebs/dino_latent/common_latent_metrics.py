"""Numpy-only common-space measurements and plots; no models or training."""
import numpy as np

from Aebs.dino_latent.proxy_comparison_math import errors, paired_summary


def generator_to_pixels(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(raw).all() or np.any(raw < -1.00001) or np.any(raw > 1.00001):
        raise ValueError("expected finite tanh generator output in [-1,1]")
    # Inverse of gen_aebs_images: images_train = 2 * images_data - 1.
    # Never per-image min/max normalization.
    return np.clip((raw + 1.) * .5, 0., 1.)


def rms_gap(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim < 1 or right.ndim < 1 or left.shape[-1] != right.shape[-1] or left.shape[-1] == 0:
        raise ValueError("vectors must have the same nonempty coordinate dimension")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite representation")
    return np.sqrt(np.mean(np.square(left-right), axis=-1))


def summarize_gaps(dino, generated, downsampled, groups, appearances, pixel_gap):
    """generated[:,0] is z=0, generated[:,1:] are ALL MC draws.

    Each gap is a nonnegative RMS coordinate distance, not a signed residual.
    `mae` in the shared summary therefore means the MEAN vector RMS distance.
    No averaging of generated vectors before taking distances.
    """
    dino, generated, downsampled = map(np.asarray, (dino, generated, downsampled))
    groups, appearances, pixel_gap = map(np.asarray, (groups, appearances, pixel_gap))
    n = len(dino)
    if (not n or dino.shape != (n,) or downsampled.shape != (n,)
            or generated.ndim != 2 or generated.shape[0] != n or generated.shape[1] < 2
            or groups.shape != (n,) or appearances.shape != (n,) or pixel_gap.shape != generated.shape):
        raise ValueError("invalid paired row/draw shapes")
    for array in (dino, generated, downsampled, pixel_gap):
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("RMS gaps must be finite and nonnegative")
    scopes = {"all_selected": np.ones(n, dtype=bool)}
    scopes.update({a: appearances == a for a in sorted(set(appearances.tolist()))})
    result = {}
    for name, mask in scopes.items():
        row = {"images": int(mask.sum()), "comparison": {}}
        for kind, sl in (("z_zero", slice(0, 1)), ("uniform_mc", slice(1, None))):
            old = generated[mask, sl]
            new = np.broadcast_to(dino[mask, None], old.shape)
            stats = paired_summary(new, old, groups[mask])
            # Scatter uses one point per image: E_z[distance], NOT distance(E_z[latent]).
            per_image_old = old.mean(axis=1)
            stats.update(
                fraction_images_dino_below_cgan_mean=float(np.mean(dino[mask] < per_image_old)),
                unique_image_count=int(mask.sum()), draws_per_image=old.shape[1],
                cgan_per_image_mean_gap=errors(per_image_old),
            )
            row["comparison"][kind] = stats
        row["downsample_reference_gap"] = errors(downsampled[mask])
        row["native_cgan_pixel_rms_uniform_mc"] = errors(pixel_gap[mask, 1:])
        result[name] = row
    return result


def make_plots(output_dir, dino, generated, downsampled, appearances, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dino, generated, downsampled, appearances = map(np.asarray, (dino, generated, downsampled, appearances))
    names = sorted(set(appearances.tolist()))
    old_mean = generated[:, 1:].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 6.4), constrained_layout=True)
    for i, name in enumerate(names):
        mask = appearances == name
        ax.scatter(old_mean[mask], dino[mask], s=12, alpha=.5,
                   color=plt.get_cmap("tab20")(i), label=name)
    limit = max(float(old_mean.max()), float(dino.max()), 1e-6) * 1.04
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel="cGAN common-latent gap (mean over all random draws)",
           ylabel="DINO q common-latent gap", title="Same image reference, same frozen 32-D space\nBelow diagonal: DINO q has smaller gap")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    path = output_dir / "01_paired_scatter.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    all_stats = summary["all_selected"]
    main = all_stats["comparison"]["uniform_mc"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, ("mae", "p95_abs", "max_abs"), ("Mean gap", "95th percentile", "Observed maximum")):
        values = [main["dino"][key], main["legacy"][key], all_stats["downsample_reference_gap"][key]]
        bars = ax.bar(["DINO q", "cGAN", "Real 32x32\nreference"], values, color=["#2878b5", "#e78b40", "#888888"])
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(max(values)*1.22, 1e-6))
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("RMS distance in shared 32-D space")
    fig.suptitle("Development data only; raw pixel error is NOT on this scale")
    fig.savefig(output_dir / "02_gap_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    y = np.arange(len(names))
    new = [summary[a]["comparison"]["uniform_mc"]["dino"]["mae"] for a in names]
    old = [summary[a]["comparison"]["uniform_mc"]["legacy"]["mae"] for a in names]
    ax.barh(y-.18, new, .36, label="DINO q", color="#2878b5")
    ax.barh(y+.18, old, .36, label="cGAN", color="#e78b40")
    ax.set(yticks=y, yticklabels=names, xlabel="Mean common-latent RMS gap",
           title="All appearances (same frozen evaluator)")
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(output_dir / "03_appearance_gaps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ["01_paired_scatter.png", "02_gap_summary.png", "03_appearance_gaps.png"]
