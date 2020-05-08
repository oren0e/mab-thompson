import numpy as np

from core import Environment, ThompsonSampler

from plot_funcs import plot_variants

if __name__ == "__main__":
    pay = np.array([0.55, 0.04, 0.80, 0.09, 0.22, 0.012])
    variants = np.arange(len(pay))

    np.random.seed(1425)
    env0 = Environment(payouts=pay, variants=variants)
    samp = ThompsonSampler(env0)
    env0.run(agent=samp, n_trials=1000)

    plot_variants(env0, iter_num=20)
    plot_variants(env0, iter_num=200)
    plot_variants(env0, iter_num=500)
    plot_variants(env0, iter_num=900)
