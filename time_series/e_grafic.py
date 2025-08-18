import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Citire fișier MDE
my_file = "./pentilfuran.MDE"
df = pd.read_csv(my_file, delim_whitespace=True, comment='#', names=["Step", "T", "E_KS", "E_tot", "Vol", "P"])

# Filtrare date (Step 1-100)
dfs = [df[df['Step'] == i].iloc[1:901] for i in range(1, 100)]
df_data = pd.concat(dfs, ignore_index=True)

steps = df_data.index

# === 1. Temperatura ===
T_values = df_data['T']
T_min, T_max = T_values.min(), T_values.max()
T_diff = T_max - T_min
print(f"[Temperatura] T_min = {T_min:.4f}, T_max = {T_max:.4f}, Diferență = {T_diff:.4f}")

scaler_T = MinMaxScaler()
T_scaled = scaler_T.fit_transform(T_values.values.reshape(-1, 1)).flatten()

# === 2. Energie ===
E_values = df_data['E_tot']
E_min, E_max = E_values.min(), E_values.max()
E_diff = E_max - E_min
print(f"[Energie] E_min = {E_min:.6f}, E_max = {E_max:.6f}, Diferență = {E_diff:.6f}")

scaler_E = MinMaxScaler()
E_scaled = scaler_E.fit_transform(E_values.values.reshape(-1, 1)).flatten()

# === Plot pentru Temperatura (două grafice) ===
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, T_values, linewidth=0.6)
plt.title(f'Temperatura T(K) - Original (ΔT = {T_diff:.4f})')
plt.xlabel('Index (proxy timp)')
plt.ylabel('T (K)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, T_scaled, color='orange', linewidth=0.6)
plt.title('Temperatura T normalizată [0,1]')
plt.xlabel('Index (proxy timp)')
plt.ylabel('T normalizat')
plt.grid(True)

plt.tight_layout()
plt.show()

# === Plot pentru Energie (două grafice) ===
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, E_values, linewidth=0.6, color='green')
plt.title(f'Energie E_tot - Original (ΔE = {E_diff:.6f})')
plt.xlabel('Index (proxy timp)')
plt.ylabel('E_tot (eV)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, E_scaled, color='red', linewidth=0.6)
plt.title('Energie E_tot normalizată [0,1]')
plt.xlabel('Index (proxy timp)')
plt.ylabel('E normalizat')
plt.grid(True)

plt.tight_layout()
plt.show()
