import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from bandit_22_greedy import *
from bandit_23_e_greedy import *
from bandit_24_optimistic_initial import *
from bandit_25_softmax import *
from bandit_26_ucb import *
from bandit_27_thompson import *


if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    all_rewards = []
    all_best = []
    all_actions = []

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 25))
    bandits.append(KAB_E_Greedy(k_arms, 0.1))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 5))
    bandits.append(KAB_Softmax(k_arms, 0.15, 0.8))

    labels = [
        'Greedy(25), ',
        'E_Greedy(0.1), ',
        'Optimistic(0.1,5), ',
        'Softmax(0.2,T,3), ',
    ]
    title = "Compare-1"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)

    runs = 1000

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 25))
    bandits.append(KAB_Softmax(k_arms, 0.15, 0.8))
    bandits.append(KAB_UCB(k_arms, 1))
    bandits.append(KAB_Thompson(k_arms, 0.5))

    labels = [
        'Greedy(25), ',
        'Softmax(0.2,T,3), ',
        'UCB(1), ',
        'Thompson(0.5), ',
    ]
    title = "Compare-2"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
