import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tensorflow.keras.optimizers import AdamW # Folosim AdamW conform sugestiei tale

# Load data from the file
my_file = "./pentilfuran.MDE"
try:
    df = pd.read_csv(my_file, delim_whitespace=True, comment='#', names=["Step", "T", "E_KS", "E_tot", "Vol", "P"])
except FileNotFoundError:
    print(f"Eroare: Fișierul '{my_file}' nu a fost găsit. Asigură-te că fișierul este în directorul corect.")
    # Creăm un DataFrame mock pentru a continua execuția și a demonstra FFT
    print("Se va folosi un DataFrame mock pentru demonstrație.")
    data_mock = {
        "Step": list(range(1, 101)) * 900, # 90k linii
        "T": np.random.rand(90000) * 50 + 1.45,
        "E_KS": np.sin(np.linspace(0, 2 * np.pi * 90000 / 50, 90000)) * 0.2 + np.random.rand(90000) * 0.01 - 2130.9,
        "E_tot": np.random.rand(90000) * 0.1 - 2130.9,
        "Vol": [3287.283] * 90000,
        "P": np.random.rand(90000) * 1 - 0.5
    }
    df = pd.DataFrame(data_mock)


# Filter and concatenate data based on 'Step'
# Versiunea optimizată - RAPIDĂ
# Asigură-te că `iloc[1:901]` este corect pentru structura datelor tale
# Dacă vrei toate rândurile din fiecare bloc, folosește `iloc[:]` sau `df[df['Step'] == i]`
dfs = [df[df['Step'] == i].iloc[1:901] for i in range(1, 100)] # Presupunem că vrei 900 de rânduri per bloc
df_data = pd.concat(dfs, ignore_index=True)

print(f"Numărul total de linii de date după filtrare și concatenare: {len(df_data)}")

# Extrage coloana E_KS (Energia K) pentru predicție
en_ks = df_data['E_KS']
# `step` ar trebui să fie indexul global al datelor concatenate, nu 'Step' din coloană
global_step_index = df_data.index.values 

# --- Ingineria Caracteristicilor: Adăugarea Frecvențelor Dominante ---
# Frecvența dominantă identificată de FFT este 0.02 Hz (cicluri per pas)
dominant_frequency = 0.02

# Calculează caracteristicile sinusoidale și cosinusoidale
# Folosim `global_step_index` pentru a asigura continuitatea fazei pe întregul set de date
df_data['sin_feature'] = np.sin(2 * np.pi * dominant_frequency * global_step_index)
df_data['cos_feature'] = np.cos(2 * np.pi * dominant_frequency * global_step_index)

# Coloanele pe care le vom scala și folosi ca intrări pentru model
features_to_scale = ['E_KS', 'sin_feature', 'cos_feature']
# Scalăm toate caracteristicile relevante
scaler = MinMaxScaler()
df_data[features_to_scale] = scaler.fit_transform(df_data[features_to_scale])

# Numărul de caracteristici de intrare pentru model
num_features = len(features_to_scale)

# Funcția pentru a crea secvențe de time-series
def create_sequences(data, sequence_length, output_steps):
    """
    Creează secvențe de intrare și ținte pentru un model de time-series.
    
    Args:
        data (np.array): Datele de intrare (caracteristici multiple).
        sequence_length (int): Lungimea secvenței de intrare.
        output_steps (int): Numărul de pași viitori de prezis.
        
    Returns:
        tuple: (sequences, targets) - array-uri numpy cu secvențele de intrare și țintele.
    """
    sequences = []
    targets = []
    # Iterează până la (lungimea datelor - lungimea secvenței de intrare - lungimea secvenței de ieșire)
    for i in range(len(data) - sequence_length - output_steps + 1):
        seq = data[i : (i + sequence_length)]
        target = data[(i + sequence_length) : (i + sequence_length + output_steps), 0] # Prezicem doar E_KS (prima coloană)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

#! Define sequence length (important parameter)
sequence_length = 200 # Lungimea secvenței de intrare
OUT_STEPS = 100       # Numărul de pași viitori de prezis

# Pregătim datele pentru funcția create_sequences
# Folosim toate caracteristicile scalate
data_for_sequences = df_data[features_to_scale].values

sequences, targets = create_sequences(data_for_sequences, sequence_length, OUT_STEPS)
print(f"Forma secvențelor de intrare: {sequences.shape}")
print(f"Forma țintelor (ieșire): {targets.shape}")

# Split data into training, validation, and test sets
# Asigurăm că split-ul este consistent cu datele tale de 90k linii
train_size = int(0.75 * len(sequences)) # Aproximativ 67000 din 89100
val_size = int(0.20 * len(sequences))   # Aproximativ 17820
test_size = len(sequences) - train_size - val_size # Restul

X_train, y_train = sequences[:train_size], targets[:train_size]
X_val, y_val = sequences[train_size : train_size + val_size], targets[train_size : train_size + val_size]
X_test, y_test = sequences[train_size + val_size :], targets[train_size + val_size :]

# Reshape pentru modelul Keras (num_samples, timesteps, num_features)
# X_train, X_val, X_test sunt deja în forma corectă datorită create_sequences
# y_train, y_val, y_test trebuie să fie reshaped pentru a se potrivi cu output-ul modelului
# Modelul prezice OUT_STEPS * num_features, dar noi vrem doar E_KS (prima caracteristică)
# Deci, ținta va fi (num_samples, OUT_STEPS, 1)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))


print(f"Forma X_train: {X_train.shape}, Y_train: {y_train.shape}")
print(f"Forma X_val: {X_val.shape}, Y_val: {y_val.shape}")
print(f"Forma X_test: {X_test.shape}, Y_test: {y_test.shape}")


def create_improved_hybrid_model(sequence_length, num_features, output_steps):
    """
    Creează un model hibrid CNN-LSTM îmbunătățit pentru predicția seriilor de timp.
    
    Args:
        sequence_length (int): Lungimea secvenței de intrare.
        num_features (int): Numărul de caracteristici de intrare.
        output_steps (int): Numărul de pași viitori de prezis.
        
    Returns:
        tf.keras.Model: Modelul Keras compilat.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(sequence_length, num_features)), # Input cu num_features
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal'), # Adăugat padding='causal'
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='causal'), # Adăugat padding='causal'
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'), # Adăugat padding='causal'
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, 
                                                          dropout=0.1, recurrent_dropout=0.1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.1, recurrent_dropout=0.1)),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Stratul de ieșire: prezice OUT_STEPS valori pentru o singură caracteristică (E_KS)
        tf.keras.layers.Dense(output_steps * 1), # *1 pentru că prezicem doar E_KS
        tf.keras.layers.Reshape([output_steps, 1]) # Reshape la (OUT_STEPS, 1)
    ])
    return model

# Inițializează modelul cu numărul corect de caracteristici
model_hybrid = create_improved_hybrid_model(sequence_length, num_features, OUT_STEPS)

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8, # Mărit răbdarea
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath="best_hybrid_model_with_fourier.keras", # Nume nou pentru fișierul modelului salvat
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7, # Factor de reducere
    patience=5, # Răbdare pentru reducerea learning rate-ului
    min_lr=1e-6, # Learning rate minim
    verbose=1
)

# Compile and Fit function
def compile_and_fit_improved(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """
    Compilează și antrenează modelul cu optimizator și callback-uri îmbunătățite.
    
    Args:
        model (tf.keras.Model): Modelul Keras de antrenat.
        X_train (np.array): Datele de antrenament (intrări).
        y_train (np.array): Țintele de antrenament.
        X_val (np.array): Datele de validare (intrări).
        y_val (np.array): Țintele de validare.
        epochs (int): Numărul maxim de epoci.
        batch_size (int): Dimensiunea batch-ului.
        
    Returns:
        tf.keras.callbacks.History: Obiectul History returnat de model.fit.
    """
    optimizer = AdamW(
        learning_rate=0.0005,  # Learning rate mai mic
        weight_decay=0.01
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Mai robust decât MSE pentru outliers
        metrics=['mae']
    )
    
    callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint
    ]
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size,  # Batch size mai mare
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history

# Train the model
print("\nÎncepe antrenarea modelului...")
history = compile_and_fit_improved(model_hybrid, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)

# Evaluează modelul pe setul de test
print("\nEvaluarea modelului pe setul de test:")
test_loss, test_mae = model_hybrid.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss (Huber): {test_loss:.6f}')
print(f'Test MAE: {test_mae:.6f}')

# Pentru a obține R2, va trebui să faci predicții și să calculezi R2 manual
from sklearn.metrics import r2_score

# Fă predicții pe setul de test
y_pred_scaled = model_hybrid.predict(X_test)

# Reshape y_test și y_pred_scaled pentru calculul R2
# Ne interesează doar prima caracteristică (E_KS) pentru predicție
y_test_flat = y_test[:, :, 0].flatten()
y_pred_flat = y_pred_scaled[:, :, 0].flatten()

# Calculează R2 score
r2 = r2_score(y_test_flat, y_pred_flat)
print(f'R2 Score: {r2:.4f}')

# --- Vizualizarea predicțiilor (opțional) ---
# Poți vizualiza o secvență de predicție vs valori reale
if len(X_test) > 0:
    sample_index = 0 # Alege o secvență din setul de test
    sample_input = X_test[sample_index]
    sample_true_output = y_test[sample_index]
    
    # Fă predicția pentru secvența selectată
    sample_prediction_scaled = model_hybrid.predict(sample_input[np.newaxis, :, :])[0]

    # Inversăm scalarea pentru a vedea valorile reale
    # Trebuie să creăm un array cu zerouri pentru sin_feature și cos_feature pentru inversare
    # și să ne asigurăm că scaler-ul a fost fit pe toate cele 3 coloane
    
    # Reconstruim un array pentru inversarea scalării
    # Valorile prezise sunt doar pentru E_KS, deci celelalte caracteristici vor fi 0 sau medii
    # Cel mai simplu este să inversăm scalarea doar pe coloana E_KS
    
    # Creăm un array temporar cu dimensiunea originală a datelor scalate (num_samples, num_features)
    # și punem predicțiile E_KS în prima coloană
    temp_pred_array = np.zeros((sample_prediction_scaled.shape[0], num_features))
    temp_pred_array[:, 0] = sample_prediction_scaled.flatten() # Punem predicțiile E_KS în prima coloană
    
    # Inversăm scalarea doar pentru coloana E_KS
    # Trebuie să folosim `inverse_transform` pe un array cu toate caracteristicile, chiar dacă celelalte sunt 0
    # O soluție mai robustă ar fi să avem un scaler separat pentru fiecare coloană sau să reconstruim datele
    
    # Pentru simplitate, vom inversa scalarea doar pe E_KS, presupunând că scaler-ul a fost fit pe E_KS ca primă coloană
    # Această parte necesită atenție la cum a fost fit scaler-ul inițial.
    # Dacă scaler-ul a fost fit pe df_data[features_to_scale], atunci trebuie să-i dăm un array de aceleași dimensiuni.
    
    # Creăm un array plin de zerouri de forma (num_samples, num_features)
    # și plasăm predicțiile noastre (care sunt doar pentru E_KS) în prima coloană
    dummy_array_pred = np.zeros((OUT_STEPS, num_features))
    dummy_array_pred[:, 0] = sample_prediction_scaled.flatten()
    
    # Inversăm scalarea pentru întregul array dummy
    sample_prediction_unscaled = scaler.inverse_transform(dummy_array_pred)[:, 0] # Luăm doar prima coloană (E_KS)

    dummy_array_true = np.zeros((OUT_STEPS, num_features))
    dummy_array_true[:, 0] = sample_true_output.flatten()
    sample_true_output_unscaled = scaler.inverse_transform(dummy_array_true)[:, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(sample_true_output_unscaled, label='Valori Reale Energie K')
    plt.plot(sample_prediction_unscaled, label='Predicții Energie K')
    plt.title('Predicție Energie K vs. Valori Reale (Exemplu)')
    plt.xlabel('Pas Viitor')
    plt.ylabel('Energie K (eV)')
    plt.legend()
    plt.grid(True)
    plt.show()
