import gym_cutting_stock
import gymnasium as gym
from core_policy import GreedyPolicy, RandomPolicy
from solvers import Greedy, Genetic

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 5

if __name__ == "__main__":
    
    # 1. Test Baseline Random Policy
    print("--- Testing Baseline Random Policy ---")
    observation, info = env.reset(seed=42)
    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Random Policy - Episode {ep}:", info)
            observation, info = env.reset(seed=ep)
            ep += 1

    # 2. Test Baseline Greedy Policy
    print("\n--- Testing Baseline Greedy Policy ---")
    observation, info = env.reset(seed=42)
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Baseline Greedy - Episode {ep}:", info)
            observation, info = env.reset(seed=ep)
            ep += 1

    # 3. Test Custom Greedy Algorithm
    print("\n--- Testing Custom Greedy Algorithm ---")
    observation, info = env.reset(seed=42)
    custom_greedy = Greedy()
    ep = 0
    while ep < NUM_EPISODES:
        action = custom_greedy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Custom Greedy - Episode {ep}:", info)
            observation, info = env.reset(seed=ep)
            ep += 1

    # 4. Test Genetic Algorithm
    print("\n--- Testing Custom Genetic Algorithm ---")
    observation, info = env.reset(seed=42)
    genetic_policy = Genetic()
    ep = 0

    while ep < NUM_EPISODES:
        action = genetic_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Genetic Algorithm - Episode {ep}:", info)
            observation, info = env.reset(seed=ep)
            ep += 1

env.close()