import matplotlib.pyplot as plt
import scienceplots

# Activate SciencePlots styles
plt.style.use(['science', 'no-latex'])

# Global rcParams adjustments (override or extend SciencePlots defaults)
plt.rcParams.update({
    # Figure
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    # Font
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    # Lines and markers
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    # Grid
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    # Legend
    'legend.fontsize': 10,
    'legend.frameon': False,
    # Tight layout
    'figure.autolayout': True,
})

# Optionally, define a helper to re‐apply style in scripts
def apply_style():
    """
    Re‐apply the default style (useful if scripts modify rcParams at runtime).
    """
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'figure.autolayout': True,
    })
