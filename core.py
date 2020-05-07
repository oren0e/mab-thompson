from __future__ import annotations

import numpy as np

from typing import Optional, Tuple, List


class Environment:
    def __init__(self, payouts: np.ndarray, variants: np.ndarray) -> None:
        self.payouts = payouts
        self.variants = variants
        self.variants_rewards: np.ndarray = np.zeros(self.variants.shape)
        self.total_reward: int = 0
        self.a_b_beta_prams: List[List[Tuple[float, float]]] = [[(1, 1) for _ in range(len(self.variants))]]

    def run(self, agent: ThompsonSampler, n_trials: int = 1000) -> Tuple[int, np.ndarray]:
        """
        Runs the simulation and return total reward from using
        the sampler chosen.
        """

        for i in range(n_trials):
            best_variant = agent.choose_variant()
            agent.reward = np.random.binomial(n=1, p=self.payouts[best_variant])    # mimick real behaviour
            agent.update()

            self.a_b_beta_prams.append([(a_i, b_i) for a_i, b_i in zip(agent.a, agent.b)])
            self.total_reward += agent.reward
            self.variants_rewards[best_variant] += agent.reward

        return self.total_reward, self.variants_rewards


class Sampler:
    def __init__(self, env: Environment) -> None:
        self.env = env
        self.theta: Optional[np.ndarray] = None
        self.best_variant: Optional[int] = None
        self.reward: int = 0

    def choose_variant(self) -> Optional[int]:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError


class ThompsonSampler(Sampler):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.a = np.ones(len(env.variants))  # prior of beta(1,1)
        self.b = np.ones(len(env.variants))  # prior of beta(1,1)

    def choose_variant(self) -> Optional[int]:
        self.theta = np.random.beta(self.a, self.b)
        self.best_variant = self.env.variants[np.argmax(self.theta)]

        return self.best_variant

    def update(self) -> None:
        self.a[self.best_variant] += self.reward
        self.b[self.best_variant] += (1 - self.reward)
