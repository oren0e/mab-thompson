import numpy as np

from scipy.stats import beta

from matplotlib import pyplot as plt

from core import Environment

from typing import Optional


def plot_variants(env0: Environment, iter_num: int, save_to: Optional[str] = None) -> None:
    x = np.linspace(0, 1, 5000)
    iteration = env0.a_b_beta_prams[iter_num]
    plt.style.use('ggplot')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(iteration))))
    for i, (a, b) in enumerate(iteration):
        c = next(color)
        y = beta.pdf(x, a=a, b=b)
        plt.plot(x, y, color=c, lw=2.1, label=f'Variant {i}')
    plt.legend()
    plt.title(f'Beta PDF, iteration number {iter_num}')
    if save_to is not None:
        plt.savefig(f'mab_iter_{iter_num}.png', format='png')
    else:
        plt.show()
