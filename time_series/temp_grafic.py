import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Citire fișier MDE
my_file = "./pentilfuran.MDE"
df = pd.read_csv(my_file, delim_whitespace=True, comment='#', names=["Step", "T", "E_KS", "E_tot", "Vol", "P"])

# Filtrare date (Step 1-100)
dfs = [df[df['Step'] == i].iloc[1:901] for i in range(1, 100)]
df_data = pd.concat(dfs, ignore_index=True)

# Extragem T
T_values = df_data['T']
steps = df_data.index

# Diferență între T_min și T_max
T_min = T_values.min()
T_max = T_values.max()
diff = T_max - T_min
print(f"T_min = {T_min:.4f}, T_max = {T_max:.4f}, Diferență = {diff:.4f}")

# Normalizare [0,1]
scaler = MinMaxScaler()
T_scaled = scaler.fit_transform(T_values.values.reshape(-1, 1)).flatten()

# === Plot două grafice unul lângă altul ===
plt.figure(figsize=(16, 6))

# Grafic 1: Valorile originale
plt.subplot(1, 2, 1)
plt.plot(steps, T_values, linewidth=0.6)
plt.title('Temperatura T(K) - Original')
plt.xlabel('Index (proxy timp)')
plt.ylabel('T (K)')
plt.grid(True)

# Grafic 2: Valorile normalizate
plt.subplot(1, 2, 2)
plt.plot(steps, T_scaled, color='orange', linewidth=0.6)
plt.title('Temperatura T normalizată [0,1]')
plt.xlabel('Index (proxy timp)')
plt.ylabel('T normalizat')
plt.grid(True)

plt.tight_layout()
plt.show()
