import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

TABLES_DIR = os.path.join('outputs', 'tables')
FIG_DIR = os.path.join('outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def _load_summary(path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    mean = df.loc['mean'] if 'mean' in df.index else df.mean(numeric_only=True)
    std = df.loc['std'] if 'std' in df.index else df.std(numeric_only=True)
    return mean, std


def plot_image_quality_summary():
    path = os.path.join(TABLES_DIR, 'image_quality_summary.csv')
    meta_path = os.path.join(TABLES_DIR, 'image_quality_summary_meta.csv')
    mean, std = _load_summary(path)

    inf_count = None
    replacement = None
    if os.path.exists(meta_path):
        try:
            meta = pd.read_csv(meta_path)
            inf_count = int(meta.get('psnr_inf_count', pd.Series([np.nan])).iloc[0])
            replacement = float(meta.get('psnr_inf_replacement', pd.Series([np.nan])).iloc[0])
        except Exception:
            inf_count = None
            replacement = None

    psnr_m = float(mean.get('psnr', np.nan))
    psnr_s = float(std.get('psnr', np.nan))
    ssim_m = float(mean.get('ssim', np.nan))
    ssim_s = float(std.get('ssim', np.nan))
    lpips_m = float(mean.get('lpips', np.nan))
    lpips_s = float(std.get('lpips', np.nan))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    ax = axes[0]
    ax.bar([0], [ssim_m], yerr=None if np.isnan(ssim_s) else [ssim_s], color='#4C78A8', width=0.6)
    ax.set_xticks([0]); ax.set_xticklabels(['SSIM'])
    ax.set_ylim(0.0, 1.0)
    ax.set_title('SSIM (↑)')
    ax.text(0, min(0.98, ssim_m + 0.05), f"{ssim_m:.3f}" + (f" ± {ssim_s:.3f}" if not np.isnan(ssim_s) else ''), ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    ax.bar([0], [lpips_m], yerr=None if np.isnan(lpips_s) else [lpips_s], color='#F58518', width=0.6)
    ax.set_xticks([0]); ax.set_xticklabels(['LPIPS'])
    ax.set_ylim(0.0, 1.0)
    ax.set_title('LPIPS (↓)')
    ax.text(0, min(0.98, lpips_m + 0.05), f"{lpips_m:.3f}" + (f" ± {lpips_s:.3f}" if not np.isnan(lpips_s) else ''), ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    ylim = (0, 60)
    if np.isfinite(psnr_m):
        ax.bar([0], [min(psnr_m, ylim[1])], yerr=None if np.isnan(psnr_s) else [psnr_s], color='#54A24B', width=0.6)
        ax.set_ylim(*ylim)
        base_label = f"{psnr_m:.1f} dB" + (f" ± {psnr_s:.1f}" if not np.isnan(psnr_s) else '')
        extra = ''
        if inf_count is not None and inf_count > 0:
            # Show how many perfect matches and what replacement was used
            if replacement is not None and np.isfinite(replacement):
                extra = f"\n(perfect={inf_count}, repl≈{replacement:.1f}dB)"
            else:
                extra = f"\n(perfect={inf_count})"
        ax.text(0, min(ylim[1]-3, (min(psnr_m, ylim[1]) + 3)), base_label + extra, ha='center', va='bottom', fontsize=9)
        ax.set_xticks([0]); ax.set_xticklabels(['PSNR'])
        ax.set_title('PSNR (dB, ↑)')
    else:
        ax.axis('off')
        text = 'PSNR: ∞ dB\n(MSE ≈ 0; images nearly/fully identical)'
        if inf_count is not None:
            text += f"\n(perfect={inf_count})"
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='#e6f4ea', edgecolor='#54A24B'))
        ax.set_title('PSNR (dB, ↑)')

    fig.suptitle('Image quality summary (mean ± std)')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = os.path.join(FIG_DIR, 'image_quality_summary_bars.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_image_quality_summary()
