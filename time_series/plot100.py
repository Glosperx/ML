import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Încarcă datele din CSV (ignoră liniile cu # dacă există)
data = pd.read_csv("Nmetil.csv", comment='#', delim_whitespace=True)

# Selectează primele 100 de rânduri
data_100 = data.head(100)

# Setează stilul pentru grafice
sns.set(style="whitegrid")

# Creează figura cu 3 grafice
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: T(K) vs Step
sns.lineplot(x='Step', y='T(K)', data=data_100, ax=axes[0], color='blue')
axes[0].set_title('Temperatura vs Step')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('T (K)')

# Plot 2: E_KS vs Step
sns.lineplot(x='Step', y='E_KS(eV)', data=data_100, ax=axes[1], color='green')
axes[1].set_title('E_KS vs Step')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('E_KS (eV)')

# Plot 3: E_tot vs Step
sns.lineplot(x='Step', y='E_tot(eV)', data=data_100, ax=axes[2], color='red')
axes[2].set_title('E_tot vs Step')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('E_tot (eV)')

plt.tight_layout()
plt.show()
