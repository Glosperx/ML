import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import random

class QLearningPathfinder:
    def __init__(self):
        # Coordonate 2D pentru orașe din România (aproximative)
        self.cities = {
            'București': (44.4268, 26.1025),
            'Cluj-Napoca': (46.7712, 23.6236),
            'Timișoara': (45.7489, 21.2087),
            'Iași': (47.1585, 27.6014),
            'Constanța': (44.1598, 28.6348),
            'Craiova': (44.3302, 23.7949),
            'Brașov': (45.6427, 25.5887),
            'Galați': (45.4353, 28.0080),
            'Ploiești': (44.9415, 26.0266),
            'Oradea': (47.0465, 21.9189)
        }
        
        self.city_list = list(self.cities.keys())
        self.n_cities = len(self.city_list)
        
        # Creează matricea de distanțe
        self.distance_matrix = self._create_distance_matrix()
        
        # Parametrii Q-Learning
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Matricea Q
        self.q_table = np.zeros((self.n_cities, self.n_cities))
        
        # Matricea de recompense (inversul distanțelor)
        self.reward_matrix = self._create_reward_matrix()
        
    def _create_distance_matrix(self):
        """Calculează distanțele euclidiene între toate orașele"""
        distances = np.zeros((self.n_cities, self.n_cities))
        
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    city1 = self.cities[self.city_list[i]]
                    city2 = self.cities[self.city_list[j]]
                    dist = np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
                    distances[i][j] = dist
                else:
                    distances[i][j] = 0
        
        return distances
    
    def _create_reward_matrix(self):
        """Creează matricea de recompense bazată pe distanțe"""
        rewards = np.zeros((self.n_cities, self.n_cities))
        max_dist = np.max(self.distance_matrix)
        
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    # Recompensă mai mare pentru distanțe mai mici
                    rewards[i][j] = (max_dist - self.distance_matrix[i][j]) / max_dist * 100
                else:
                    rewards[i][j] = -100  # Penalizare pentru a rămâne în același oraș
        
        return rewards
    
    def get_valid_actions(self, state, visited):
        """Returnează acțiunile valide (orașele nevizitate)"""
        return [i for i in range(self.n_cities) if i not in visited]
    
    def choose_action(self, state, valid_actions):
        """Alege acțiunea folosind epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            # Explorare
            return random.choice(valid_actions)
        else:
            # Exploatare
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def train(self, episodes=1000, start_city='București', end_city='Iași'):
        """Antrenează agentul Q-Learning"""
        start_idx = self.city_list.index(start_city)
        end_idx = self.city_list.index(end_city)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            current_state = start_idx
            visited = {start_idx}
            path = [start_idx]
            total_reward = 0
            steps = 0
            max_steps = self.n_cities * 2  # Prevenir ciclurile infinite
            
            while current_state != end_idx and steps < max_steps:
                valid_actions = self.get_valid_actions(current_state, visited)
                
                if not valid_actions or end_idx not in valid_actions and len(visited) > self.n_cities // 2:
                    # Dacă nu mai sunt acțiuni valide, permite să meargă la destinație
                    valid_actions = [end_idx]
                
                if not valid_actions:
                    break
                
                action = self.choose_action(current_state, valid_actions)
                
                # Calculează recompensa
                reward = self.reward_matrix[current_state][action]
                
                # Bonus suplimentar pentru a ajunge la destinație
                if action == end_idx:
                    reward += 200
                
                # Update Q-table
                future_rewards = []
                next_valid_actions = self.get_valid_actions(action, visited | {action})
                if next_valid_actions:
                    future_rewards = [self.q_table[action][next_action] for next_action in next_valid_actions]
                
                max_future_reward = max(future_rewards) if future_rewards else 0
                
                self.q_table[current_state][action] = (
                    self.q_table[current_state][action] + 
                    self.alpha * (reward + self.gamma * max_future_reward - self.q_table[current_state][action])
                )
                
                current_state = action
                visited.add(action)
                path.append(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return episode_rewards, episode_lengths
    
    def find_best_path(self, start_city='București', end_city='Iași'):
        """Găsește cel mai bun drum folosind Q-table antrenat"""
        start_idx = self.city_list.index(start_city)
        end_idx = self.city_list.index(end_city)
        
        current_state = start_idx
        visited = {start_idx}
        path = [start_idx]
        total_distance = 0
        
        while current_state != end_idx and len(path) < self.n_cities:
            valid_actions = self.get_valid_actions(current_state, visited)
            
            if not valid_actions:
                valid_actions = [end_idx]
            
            if end_idx in valid_actions:
                valid_actions = [end_idx]  # Mergi direct la destinație dacă e posibil
            
            # Alege acțiunea cu Q-value maxim
            q_values = [self.q_table[current_state][action] for action in valid_actions]
            best_action = valid_actions[np.argmax(q_values)]
            
            total_distance += self.distance_matrix[current_state][best_action]
            current_state = best_action
            visited.add(best_action)
            path.append(best_action)
        
        return path, total_distance
    
    def plot_results(self, episode_rewards, episode_lengths, best_path, total_distance):
        """Plotează rezultatele"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Recompensele pe episoade
        ax1.plot(episode_rewards)
        ax1.set_title('Recompense pe Episoade')
        ax1.set_xlabel('Episod')
        ax1.set_ylabel('Recompensă Totală')
        ax1.grid(True)
        
        # 2. Lungimea drumurilor pe episoade
        ax2.plot(episode_lengths)
        ax2.set_title('Lungimea Drumurilor pe Episoade')
        ax2.set_xlabel('Episod')
        ax2.set_ylabel('Numărul de Pași')
        ax2.grid(True)
        
        # 3. Harta cu orașele și drumul optim
        ax3.set_title(f'Drumul Optim (Distanța: {total_distance:.2f})')
        
        # Plotează toate orașele
        for city, coords in self.cities.items():
            ax3.scatter(coords[1], coords[0], s=100, c='lightblue', edgecolor='black')
            ax3.annotate(city, (coords[1], coords[0]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Plotează drumul optim
        path_coords = []
        for i in best_path:
            city_name = self.city_list[i]
            coords = self.cities[city_name]
            path_coords.append(coords)
        
        if len(path_coords) > 1:
            path_coords = np.array(path_coords)
            ax3.plot(path_coords[:, 1], path_coords[:, 0], 'r-', linewidth=2, alpha=0.7)
            
            # Marchează startul și sfârșitul
            ax3.scatter(path_coords[0, 1], path_coords[0, 0], s=200, c='green', 
                       marker='s', label='Start', edgecolor='black')
            ax3.scatter(path_coords[-1, 1], path_coords[-1, 0], s=200, c='red', 
                       marker='s', label='Sfârșit', edgecolor='black')
        
        ax3.set_xlabel('Longitudine')
        ax3.set_ylabel('Latitudine')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-table heatmap
        im = ax4.imshow(self.q_table, cmap='viridis', aspect='auto')
        ax4.set_title('Q-Table Heatmap')
        ax4.set_xlabel('Acțiuni (Orașul de destinație)')
        ax4.set_ylabel('Stări (Orașul curent)')
        
        # Adaugă labels pentru orașe
        ax4.set_xticks(range(self.n_cities))
        ax4.set_yticks(range(self.n_cities))
        ax4.set_xticklabels([city[:4] for city in self.city_list], rotation=45)
        ax4.set_yticklabels([city[:4] for city in self.city_list])
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.show()
        
        # Afișează drumul în text
        print(f"\n{'='*50}")
        print("DRUMUL OPTIM GĂSIT:")
        print(f"{'='*50}")
        for i, city_idx in enumerate(best_path):
            city_name = self.city_list[city_idx]
            if i < len(best_path) - 1:
                next_city_idx = best_path[i + 1]
                distance = self.distance_matrix[city_idx][next_city_idx]
                print(f"{i+1}. {city_name} → {self.city_list[next_city_idx]} ({distance:.2f} km)")
            else:
                print(f"{i+1}. {city_name} (DESTINAȚIE)")
        
        print(f"\nDistanța totală: {total_distance:.2f} km")
        print(f"Numărul de orașe vizitate: {len(best_path)}")

# Rulează exemplul
if __name__ == "__main__":
    print("🚗 Q-Learning pentru găsirea drumului optim între orașe din România")
    print("=" * 60)
    
    # Creează și antrenează agentul
    pathfinder = QLearningPathfinder()
    
    start_city = 'București'
    end_city = 'Cluj-Napoca'
    
    print(f"Antrenez agentul pentru drumul {start_city} → {end_city}...")
    
    # Antrenează
    episode_rewards, episode_lengths = pathfinder.train(
        episodes=2000, 
        start_city=start_city, 
        end_city=end_city
    )
    
    # Găsește cel mai bun drum
    best_path, total_distance = pathfinder.find_best_path(start_city, end_city)
    
    # Plotează rezultatele
    pathfinder.plot_results(episode_rewards, episode_lengths, best_path, total_distance)