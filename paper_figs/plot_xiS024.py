import numpy as np
import h5py
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{physics}"
params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)

file_data = h5py.File("./data/xiSgcs_GR_LOWZ_z0.25.hdf5", "r")
file_theory = h5py.File("./data/theory_xiSgcs_GR_LOWZ_z0.25.hdf5", "r")
f_c = file_theory.attrs["f_c"]
f_s = file_theory.attrs["f_s"]
s_max = 100.0
fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(
    1,
    2,
    wspace=0.23,
    width_ratios=[1, 1],
)
ax00 = plt.subplot(gs[0, 0])
ax01 = plt.subplot(gs[0, 1])


pairs = "gg"
_lst = []
for ibox in range(1, 100 + 1):
    group_data = file_data[f"box{ibox}"]
    mask = group_data[f"s_bincentre_{pairs}"][...] < s_max
    ss_sim = group_data[f"s_bincentre_{pairs}"][mask]
    xx = np.vstack(
        (
            group_data[f"xiS0_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
        )
    ).T
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0],
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0],
    lw=0,
    marker=".",
    markersize=5,
    elinewidth=0.5,
    color="k",
    label=r"$\mathrm{total}$",
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1],
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1],
    lw=0,
    marker=".",
    markersize=5,
    elinewidth=0.5,
    color="k",
    label=r"$\mathrm{total}$",
)

mask = file_theory["s_bincentre"][...] < s_max
ss_theory = file_theory["s_bincentre"][mask]
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_{pairs}"][mask],
    lw=1.5,
    color="k",
    label=r"$\mathrm{theory}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_{pairs}"][mask],
    lw=1.5,
    color="k",
    label=r"$\mathrm{theory}$",
)


pairs = "cc"
_lst = []
for ibox in range(1, 100 + 1):
    group_data = file_data[f"box{ibox}"]
    mask = group_data[f"s_bincentre_{pairs}"][...] < s_max
    ss_sim = group_data[f"s_bincentre_{pairs}"][mask]
    xx = np.vstack(
        (
            group_data[f"xiS0_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
        )
    ).T
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * f_c**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * f_c**2,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="r",
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * f_c**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * f_c**2,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="r",
)


mask = file_theory["s_bincentre"][...] < s_max
ss_theory = file_theory["s_bincentre"][mask]
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_2h_{pairs}"][mask] * f_c**2,
    lw=0.7,
    color="r",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_2h_{pairs}"][mask] * f_c**2,
    lw=0.7,
    color="r",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)


pairs = "cs"
_lst = []
for ibox in range(1, 100 + 1):
    group_data = file_data[f"box{ibox}"]
    mask = group_data[f"s_bincentre_{pairs}"][...] < s_max
    ss_sim = group_data[f"s_bincentre_{pairs}"][mask]
    xx = np.vstack(
        (
            group_data[f"xiS0_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
        )
    ).T
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)

ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * 2 * f_c * f_s,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * 2 * f_c * f_s,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="mediumseagreen",
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * 2 * f_c * f_s,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * 2 * f_c * f_s,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="mediumseagreen",
)


mask = file_theory["s_bincentre"][...] < s_max
ss_theory = file_theory["s_bincentre"][mask]
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_1h_{pairs}"][mask] * 2 * f_c * f_s,
    lw=1.0,
    linestyle="--",
    color="mediumseagreen",
    label=r"$\mathrm{1h}\text{-}\mathrm{" + pairs + r"}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_1h_{pairs}"][mask] * 2 * f_c * f_s,
    lw=1.0,
    linestyle="--",
    color="mediumseagreen",
    label=r"$\mathrm{1h}\text{-}\mathrm{" + pairs + r"}$",
)
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_2h_{pairs}"][mask] * 2 * f_c * f_s,
    lw=1.0,
    color="mediumseagreen",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_2h_{pairs}"][mask] * 2 * f_c * f_s,
    lw=1.0,
    color="mediumseagreen",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)


pairs = "ss"
_lst = []
for ibox in range(1, 100 + 1):
    group_data = file_data[f"box{ibox}"]
    mask = group_data[f"s_bincentre_{pairs}"][...] < s_max
    ss_sim = group_data[f"s_bincentre_{pairs}"][mask]
    xx = np.vstack(
        (
            group_data[f"xiS0_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
            group_data[f"xiS2_{pairs}"][mask],
        )
    ).T
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)

ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * f_s**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * f_s**2,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="mediumblue",
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * f_s**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * f_s**2,
    lw=0,
    marker="o",
    markersize=1.5,
    elinewidth=0.0,
    color="mediumblue",
)


mask = file_theory["s_bincentre"][...] < s_max
ss_theory = file_theory["s_bincentre"][mask]


ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_1h_{pairs}"][mask] * f_s**2,
    lw=1.0,
    linestyle="--",
    color="mediumblue",
    label=r"$\mathrm{1h}\text{-}\mathrm{" + pairs + r"}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_1h_{pairs}"][mask] * f_s**2,
    lw=1.0,
    linestyle="--",
    color="mediumblue",
    label=r"$\mathrm{1h}\text{-}\mathrm{" + pairs + r"}$",
)
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS0_2h_{pairs}"][mask] * f_s**2,
    lw=1.0,
    color="mediumblue",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f"xiS2_2h_{pairs}"][mask] * f_s**2,
    lw=1.0,
    color="mediumblue",
    label=r"$\mathrm{2h}\text{-}\mathrm{" + pairs + r"}$",
)


ax00.set_xlim([0.9, s_max])
ax01.set_xlim([0.9, s_max])

ax00.set_xlabel(
    r"$s / (h^{-1}\mathrm{Mpc})$",
    fontsize=16,
)
ax01.set_xlabel(
    r"$s / (h^{-1}\mathrm{Mpc})$",
    fontsize=16,
)
ax00.set_ylabel(
    r"$s^2 \xi^{\mathrm{S}}_0(s)$",
    fontsize=16,
)
ax01.set_ylabel(
    r"$s^2 \xi^{\mathrm{S}}_2(s)$",
    fontsize=16,
)
ax01.legend(
    frameon=False,
    ncols=2,
    fontsize=12,
)
# ax01.set_xscale("log")
# ax00.set_xscale("log")

plt.savefig(
    "./figs/xiS024_z0.25.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)


file_data.close()
file_theory.close()
