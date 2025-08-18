import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler # Importăm MinMaxScaler

# Load data from the file
# Asumi că fișierul 'pentilfuran.MDE' este disponibil în același director
my_file = "./pentilfuran.MDE"
try:
    df = pd.read_csv(my_file, delim_whitespace=True, comment='#', names=["Step", "T", "E_KS", "E_tot", "Vol", "P"])
except FileNotFoundError:
    print(f"Eroare: Fișierul '{my_file}' nu a fost găsit. Asigură-te că fișierul este în directorul corect.")
    # Creăm un DataFrame mock pentru a continua execuția și a demonstra FFT
    print("Se va folosi un DataFrame mock pentru demonstrație.")
    data_mock = {
        "Step": list(range(1, 101)) * 5,
        "T": np.random.rand(500) * 50 + 1.45,
        "E_KS": np.sin(np.linspace(0, 4 * np.pi, 500)) * 0.1 + np.random.rand(500) * 0.01 - 2130.9,
        "E_tot": np.random.rand(500) * 0.1 - 2130.9,
        "Vol": [3287.283] * 500,
        "P": np.random.rand(500) * 1 - 0.5
    }
    df = pd.DataFrame(data_mock)
    # Ajustăm E_KS pentru a simula ciclicitatea pe care o așteptăm
    # Adăugăm o componentă sinusoidală mai pronunțată
    time_points = np.arange(len(df))
    # Simulează o oscilație cu o perioadă de aproximativ 50 de pași
    df['E_KS'] = -2130.9 + 0.2 * np.sin(2 * np.pi * time_points / 50) + 0.05 * np.random.randn(len(df))


# Filter and concatenate data based on 'Step'
# Folosim varianta optimizată pentru a procesa datele
dfs = [df[df['Step'] == i].iloc[1:901] for i in range(1, 100)]
df_data = pd.concat(dfs, ignore_index=True)

print(f"Numărul total de linii de date după filtrare și concatenare: {len(df_data)}")

# Extrage coloana E_KS pentru analiza FFT
# Conform descrierii tale, E_KS este energia care variază sinusoidal
e_ks_data = df_data['E_KS'].values.reshape(-1, 1) # Reshape pentru MinMaxScaler

# --- Normalizarea datelor E_KS ---
scaler = MinMaxScaler()
e_ks_data_scaled = scaler.fit_transform(e_ks_data).flatten() # Normalizăm și aplatizăm înapoi

# --- Analiza FFT ---
# Numărul de puncte din semnal
N = len(e_ks_data_scaled)

# Frecvența de eșantionare (numărul de eșantioane pe unitatea de timp).
# Deoarece 'Step' este un index și nu avem o unitate de timp explicită,
# vom asuma că pasul dintre eșantioane este 1.
# Dacă știm intervalul de timp real între pași, ar trebui să-l folosim aici.
Fs = 1.0 # Frecvența de eșantionare (1 eșantion pe "pas")

# Aplică Transformata Fourier Rapidă (FFT) pe datele normalizate
yf = fft(e_ks_data_scaled)

# Calculează frecvențele corespunzătoare
xf = fftfreq(N, 1 / Fs)

# Calculează magnitudinea (amplitudinea) transformatei Fourier
# Luăm doar prima jumătate, deoarece spectrul este simetric pentru semnale reale
magnitude = 2.0/N * np.abs(yf[0:N//2])
frequencies = xf[0:N//2]

# Găsește frecvențele dominante
# Excludem frecvența zero (componenta DC) care reprezintă valoarea medie
# și care nu ne interesează pentru oscilații
dominant_indices = np.where(frequencies > 0)[0]
if len(dominant_indices) > 0:
    # Sortăm după magnitudinea descrescătoare pentru a găsi cele mai puternice frecvențe
    sorted_indices = dominant_indices[np.argsort(magnitude[dominant_indices])][::-1]
    
    print("\nFrecvențe dominante (primele 5):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"Frecvență: {frequencies[idx]:.6f} Hz, Amplitudine: {magnitude[idx]:.6f}")
else:
    print("\nNu s-au găsit frecvențe dominante semnificative (în afara componentei DC).")


# --- Vizualizare ---
plt.figure(figsize=(14, 8))

# Plot semnalul original E_KS (normalizat)
plt.subplot(2, 1, 1)
plt.plot(e_ks_data_scaled)
plt.title('Evoluția Energiei K (Normalizată) în Timp')
plt.xlabel('Pas')
plt.ylabel('Energie K (Normalizată)')
plt.grid(True)

# Plot spectrul de frecvență
plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude)
plt.title('Spectrul de Frecvență al Energiei K (Normalizată)')
plt.xlabel('Frecvență (Hz)') # Aici "Hz" se referă la "cicluri pe pas"
plt.ylabel('Amplitudine')
plt.grid(True)
plt.xlim(0, Fs / 2) # Afișăm doar frecvențele pozitive până la frecvența Nyquist
plt.tight_layout()
plt.show()

print("\nAnaliza FFT a fost realizată pe datele normalizate. Graficele de mai sus arată semnalul Energie K și spectrul său de frecvență.")
print("Vârfurile din spectrul de frecvență indică frecvențele dominante ale oscilațiilor energiei.")
