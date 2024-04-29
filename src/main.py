import gymnasium
import numpy as np
import __init__
import matplotlib.pyplot as plt

# def test_bandit_slots():
"""
Tests that the MultiArmedBandit implementation successfully finds the slot
machine with the largest expected reward.
"""
from multi_armed_bandit import MultiArmedBandit
# from np.random import rng
# rng.seed()

env = gymnasium.make('SlotMachines-v0',mean_list=[0.1,0.05],std_dev=np.sqrt([0.1,0.3]))
env.seed(0)
means = np.array([m.mean for m in env.machines])
print(means)

agent = MultiArmedBandit(epsilon=0.1)
total_sum_list=[]
for _ in range(100):
    state_action_values,rewards,total_sum = agent.fit(env, steps=100)
    total_sum_list.append(total_sum)
print(total_sum_list*1000)
print(np.mean(total_sum))
trails=range(1,101)
total_sum_list=np.array(total_sum_list)
plt.plot(trails,1000*total_sum_list-1000)
plt.xlabel("trials")
plt.ylabel("total return after 100 days/$")
plt.title("Total return after 100 days versus trials plot")
print(f"The average of the return is {np.mean(1000*total_sum_list-1000)}")
print(f"The variance of the return is {np.var(1000*total_sum_list-1000)} ")
print("success")