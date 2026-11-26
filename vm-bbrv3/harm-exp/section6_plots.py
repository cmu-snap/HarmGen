"""Plots for the Section 6 comparisons."""

import matplotlib.pyplot as plt


STYLES = {
    "harmgen": ("HarmGen", "#C0392B", "-", ""),
    "random-search": ("Random search", "#3B6FB6", "--", "//"),
    "hill-climbing": ("Hill climbing", "#2A786C", ":", ".."),
    "parameter-sweep": ("Parameter sweep", "#6E6E6E", "-.", "xx"),
}


def draw_cdf(ax, harms, **style):
    x = sorted(harms)
    y = [rank / len(x) for rank in range(len(x) + 1)]
    ax.step([x[0], *x], y, where="post", linewidth=1.6, **style)


def plot_random_pairs(output, distributions):
    styles = (
        ("#0072B2", "-"), ("#009E73", "--"), ("#D55E00", "-."),
        ("#CC79A7", (0, (5, 2, 1, 2, 1, 2))), ("#666666", ":"),
    )
    with plt.style.context("default"), plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.8, 3.1), layout="constrained")
        for (label, harms), (color, linestyle) in zip(distributions.items(), styles, strict=True):
            draw_cdf(ax, harms, label=label, color=color, linestyle=linestyle)
        ax.set(xlabel="Relative harm", ylabel="Fraction of random scenarios")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2, frameon=False)
        fig.savefig(output / "figure12_relative_harm_cdf.pdf")
        plt.close(fig)


def plot(output, top_harms, distances, budgets):
    # The shared tc runner sets a global style for per-flow plots.
    with plt.style.context("default"), plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.8, 3.1), layout="constrained")
        for method, harms in top_harms.items():
            name, color, linestyle, _ = STYLES[method]
            draw_cdf(ax, harms, label=f"{name} ({budgets[method]:,})",
                     color=color, linestyle=linestyle)
        top_n = len(top_harms["harmgen"])
        ax.set(xlabel="Relative harm", ylabel=f"Fraction of top {top_n} scenarios")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2, frameon=False,
                  title="Method (measured scenarios)")
        fig.savefig(output / "figure13_relative_harm_cdf.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 3.1), layout="constrained")
        boxes = ax.boxplot(list(distances.values()), widths=0.55, patch_artist=True,
                           medianprops={"color": "black"})
        for method, box in zip(distances, boxes["boxes"]):
            _, color, _, hatch = STYLES[method]
            box.set(facecolor="white", edgecolor=color, hatch=hatch, linewidth=1.4)
        labels = [STYLES[name][0].replace(" ", "\n", 1) for name in distances]
        ax.set_xticks(range(1, len(distances) + 1), labels)
        ax.set_ylabel("Normalized Euclidean distance")
        fig.savefig(output / "figure14_scenario_diversity.pdf")
        plt.close(fig)
