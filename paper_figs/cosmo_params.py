import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# ---------------------------
# Matplotlib / seaborn setup
# ---------------------------
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.size': 24,
    'axes.titlesize': 18,
    'axes.labelsize': 26,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{physics}',
})

# ---------------------------
# Load data
# ---------------------------
cosmos = np.loadtxt('./wide_sample_first_64.txt', skiprows=1)

df_train = pd.DataFrame({
    'Om0': cosmos[1:54, 0],
    'h':   cosmos[1:54, 1],
    'S8':  cosmos[1:54, 3],
    'ns':  cosmos[1:54, 2],
    "dataset": r"$\mathrm{train}$",
})

df_test = pd.DataFrame({
    'Om0': cosmos[54:-1, 0],
    'h':   cosmos[54:-1, 1],
    'S8':  cosmos[54:-1, 3],
    'ns':  cosmos[54:-1, 2],
    "dataset": r"$\mathrm{test}$",
})

df_fid = pd.DataFrame({
    'Om0': [0.3089],
    'h':   [0.6774],
    'S8':  [0.8159 * np.sqrt(0.3089 / 0.3)],
    'ns':  [0.9667],
    "dataset": "fiducial",
})

# Combine datasets (train + test, fiducial can be plotted separately if needed)
df = pd.concat((df_test, df_train), ignore_index=True)

# Rename columns for LaTeX labels
df = df.rename(columns={
    'Om0': r'$\Omega_{\mathrm{m}0}$',
    'h':   r'$h$',
    'S8':  r'$S_{8}$',
    'ns':  r'$n_{\mathrm{s}}$',
})

# ---------------------------
# Pairplot
# ---------------------------
pair = sns.pairplot(
    df,
    diag_kind="hist",
    corner=True,
    plot_kws=dict(linewidth=0.1),
    height=1.8,
    hue="dataset",
    palette=['red', 'k'],
    markers=["D", "."],
)

# Adjust spacing
pair.fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Move legend
sns.move_legend(
    pair,
    "upper left",
    bbox_to_anchor=(.55, .90),
    title=None,
    frameon=True,
    fontsize=26,
)

# Save figure
pair.fig.savefig(
    "./figs/cosmoparams_lcdm.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)
