import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
import matplotlib.pyplot as plt
import time

# Configurare
with open("romania_cities_with_costs.json", "r") as f:
    data = json.load(f)

cities = data["cities"]
time_matrix = np.array(data["time_matrix"])
penalty_matrix = np.array(data["penalty_matrix"])
num_cities = len(cities)

class TSPEnv(gym.Env):
    def __init__(self):
        super(TSPEnv, self).__init__()
        self.action_space = spaces.Discrete(num_cities)
        self.observation_space = spaces.Dict({
            'current_city': spaces.Discrete(num_cities),
            'visited': spaces.MultiBinary(num_cities)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_city = 0
        self.visited = np.zeros(num_cities, dtype=np.int32)
        self.visited[0] = 1
        self.visited_count = 1
        self.total_reward = 0
        info = {}
        return self._get_observation(), info

    def _get_observation(self):
        return {'current_city': self.current_city, 'visited': self.visited.copy()}

    def step(self, action):
        done = False
        truncated = False
        reward = 0
        
        if self.visited[action] == 0:
            time_cost = time_matrix[self.current_city][action]
            penalty_cost = penalty_matrix[self.current_city][action]
            reward = -(time_cost + penalty_cost)
            self.total_reward += reward
            
            self.current_city = action
            self.visited[action] = 1
            self.visited_count += 1
            
            if self.visited_count == num_cities:
                done = True
                # Adaugă costul de întoarcere la punctul de start
                return_cost = time_matrix[self.current_city][0] + penalty_matrix[self.current_city][0]
                reward -= return_cost
                self.total_reward -= return_cost
        else:
            reward = -1000  # Penalizare mare pentru acțiune invalidă
            self.total_reward += reward
            done = True

        info = {}
        return self._get_observation(), reward, done, truncated, info

# Crearea mediului
env = TSPEnv()
train_env = tf_py_environment.TFPyEnvironment(env)

# Rețele neuronale
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(128, 128),
    activation_fn=tf.keras.activations.relu
)

value_net = value_network.ValueNetwork(
    train_env.observation_spec(),
    fc_layer_params=(128, 128),
    activation_fn=tf.keras.activations.relu
)

# Agent PPO
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=value_net,
    train_step_counter=train_step_counter,
    entropy_regularization=0.01,
    importance_ratio_clipping=0.2,
    discount_factor=0.99,
    gradient_clipping=1.0,
    value_pred_loss_coef=0.5,
    num_epochs=10
)
agent.initialize()

# Vizualizare în timp real
plt.ion()
fig, ax = plt.subplots(figsize=(12, 10))

def plot_route(observation, total_reward, episode):
    ax.clear()
    current = observation['current_city']
    visited = observation['visited']
    
    # Plotează toate orașele
    lats = [city['lat'] for city in cities]
    lons = [city['lon'] for city in cities]
    names = [city['capitala'] for city in cities]
    
    ax.scatter(lons, lats, c='blue', s=50)
    
    # Adaugă numele orașelor
    for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
        ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Evidențiază orașele vizitate
    visited_indices = np.where(visited == 1)[0]
    ax.scatter(np.array(lons)[visited_indices], 
               np.array(lats)[visited_indices], 
               c='red', s=100, marker='s')
    
    # Evidențiază orașul curent
    ax.scatter(lons[current], lats[current], c='green', s=150, marker='*')
    
    # Conectează traseul
    if len(visited_indices) > 1:
        for i in range(len(visited_indices)-1):
            start_city = visited_indices[i]
            end_city = visited_indices[i+1]
            ax.plot([lons[start_city], lons[end_city]], 
                    [lats[start_city], lats[end_city]], 
                    'k-', alpha=0.7, linewidth=2)
    
    ax.set_title(f"Episod: {episode}, Recompensă Totală: {total_reward:.2f}")
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    plt.draw()
    plt.pause(0.01)

# Antrenare
num_iterations = 500
best_reward = float('-inf')
best_route = None

# Calcul Nearest Neighbor pentru comparație
def nearest_neighbor_heuristic():
    visited = [False] * num_cities
    current = 0
    visited[0] = True
    total_cost = 0
    route = [0]
    
    for _ in range(num_cities - 1):
        min_cost = float('inf')
        next_city = -1
        
        for j in range(num_cities):
            if not visited[j]:
                cost = time_matrix[current][j] + penalty_matrix[current][j]
                if cost < min_cost:
                    min_cost = cost
                    next_city = j
        
        total_cost += min_cost
        current = next_city
        visited[current] = True
        route.append(current)
    
    # Adaugă costul de întoarcere
    total_cost += time_matrix[current][0] + penalty_matrix[current][0]
    return total_cost, route

nn_cost, nn_route = nearest_neighbor_heuristic()
print(f"Costul heuristicii Nearest Neighbor: {nn_cost:.2f}")

for episode in range(num_iterations):
    time_step = train_env.reset()
    episode_return = 0.0
    
    while not time_step.is_last():
        action_step = agent.collect_policy.action(time_step)
        next_time_step = train_env.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        
        # Antrenează agentul
        experience = ts.transition(time_step, action_step.action, next_time_step)
        agent.train(experience)
        
        time_step = next_time_step
        
        # Vizualizare (doar la fiecare 10 episoade pentru performanță)
        if episode % 10 == 0:
            plot_route(env._get_observation(), env.total_reward, episode)
    
    # Salvează cel mai bun traseu
    if env.total_reward > best_reward:
        best_reward = env.total_reward
        best_route = {
            'cities_visited': [cities[i]['capitala'] for i in np.where(env.visited == 1)[0]],
            'total_cost': -env.total_reward,
            'efficiency_ratio': (-env.total_reward / nn_cost) * 100
        }
    
    if episode % 50 == 0:
        print(f"Episod {episode}, Recompensă: {env.total_reward:.2f}, Cel mai bun: {best_reward:.2f}")

plt.ioff()

# Salvare rezultate
with open("tsp_solution.json", "w", encoding="utf-8") as f:
    json.dump(best_route, f, ensure_ascii=False, indent=4)

print("Antrenament completat. Rezultatele au fost salvate în tsp_solution.json")
print(f"Cost total: {best_route['total_cost']:.2f}")
print(f"Procent față de Nearest Neighbor: {best_route['efficiency_ratio']:.2f}%")

# Afișează traseul final
plt.figure(figsize=(12, 10))
lats = [city['lat'] for city in cities]
lons = [city['lon'] for city in cities]
names = [city['capitala'] for city in cities]

plt.scatter(lons, lats, c='blue', s=50)
for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
    plt.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Traseul optim
visited_indices = np.where(env.visited == 1)[0]
plt.scatter(np.array(lons)[visited_indices], 
           np.array(lats)[visited_indices], 
           c='red', s=100, marker='s')

for i in range(len(visited_indices)-1):
    start_city = visited_indices[i]
    end_city = visited_indices[i+1]
    plt.plot([lons[start_city], lons[end_city]], 
            [lats[start_city], lats[end_city]], 
            'k-', alpha=0.7, linewidth=2)

plt.title(f"Traseul optim - Cost: {best_route['total_cost']:.2f}")
plt.xlabel("Longitudine")
plt.ylabel("Latitudine")
plt.savefig("optimal_route.png")
plt.show()