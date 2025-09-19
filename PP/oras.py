import json
import numpy as np

# Încarcă orașele din fișier sau direct în cod
cities = [
    {"capitala": "Alba Iulia", "lat": 45.8666, "lon": 23.5666},
    {"capitala": "Arad", "lat": 46.1833, "lon": 21.3167},
    {"capitala": "Pitești", "lat": 44.8667, "lon": 24.8667},
    {"capitala": "Bacău", "lat": 46.5667, "lon": 26.9167},
    {"capitala": "Oradea", "lat": 47.0667, "lon": 21.9333},
    {"capitala": "Bistrița", "lat": 47.1000, "lon": 24.5000},
    {"capitala": "Brașov", "lat": 45.6500, "lon": 25.5833},
    {"capitala": "Brăila", "lat": 45.2667, "lon": 27.9667},
    {"capitala": "Buzău", "lat": 45.1500, "lon": 26.8167},
    {"capitala": "Reșița", "lat": 45.3000, "lon": 22.9000},
    {"capitala": "Cluj-Napoca", "lat": 46.7833, "lon": 23.6000},
    {"capitala": "Constanța", "lat": 44.1833, "lon": 28.6333},
    {"capitala": "Sfântu Gheorghe", "lat": 45.8833, "lon": 25.7833},
    {"capitala": "Târgoviște", "lat": 44.9333, "lon": 25.4500},
    {"capitala": "Craiova", "lat": 44.3333, "lon": 23.8167},
    {"capitala": "Galați", "lat": 45.4333, "lon": 28.0500},
    {"capitala": "Giurgiu", "lat": 43.9000, "lon": 25.9833},
    {"capitala": "Târgu Jiu", "lat": 45.0333, "lon": 23.2833},
    {"capitala": "Miercurea Ciuc", "lat": 46.3667, "lon": 25.8000},
    {"capitala": "Deva", "lat": 45.8833, "lon": 22.9333},
    {"capitala": "Slobozia", "lat": 44.5667, "lon": 27.3667},
    {"capitala": "Iași", "lat": 47.1667, "lon": 27.6000},
    {"capitala": "Buftea", "lat": 44.5333, "lon": 25.9500},
    {"capitala": "Baia Mare", "lat": 47.6500, "lon": 23.5833},
    {"capitala": "Drobeta Turnu Severin", "lat": 44.6167, "lon": 22.9833},
    {"capitala": "Târgu Mureș", "lat": 46.5500, "lon": 24.5500},
    {"capitala": "Piatra Neamț", "lat": 46.9333, "lon": 26.3667},
    {"capitala": "Slatina", "lat": 44.4333, "lon": 24.3667},
    {"capitala": "Ploiești", "lat": 44.9333, "lon": 26.0333},
    {"capitala": "Zalău", "lat": 47.2000, "lon": 23.0500},
    {"capitala": "Satu Mare", "lat": 47.7833, "lon": 22.8833},
    {"capitala": "Sibiu", "lat": 45.8000, "lon": 24.1500},
    {"capitala": "Suceava", "lat": 47.6500, "lon": 26.2500},
    {"capitala": "Alexandria", "lat": 44.1833, "lon": 25.3667},
    {"capitala": "Timișoara", "lat": 45.7500, "lon": 21.2167},
    {"capitala": "Tulcea", "lat": 45.1667, "lon": 28.7833},
    {"capitala": "Vaslui", "lat": 46.2500, "lon": 27.7333},
    {"capitala": "Râmnicu Vâlcea", "lat": 45.1000, "lon": 24.3667},
    {"capitala": "Focșani", "lat": 45.7000, "lon": 27.1833},
    {"capitala": "București", "lat": 44.4268, "lon": 26.1025}
]

n = len(cities)
time_matrix = np.zeros((n, n))
penalty_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            # Timp = distanță euclidiană
            lat_diff = cities[i]["lat"] - cities[j]["lat"]
            lon_diff = cities[i]["lon"] - cities[j]["lon"]
            time_matrix[i, j] = np.sqrt(lat_diff**2 + lon_diff**2)
            # Penalizare = random între 0 și 10
            penalty_matrix[i, j] = np.random.rand() * 10

# Salvare JSON complet
data = {
    "cities": cities,
    "time_matrix": time_matrix.tolist(),
    "penalty_matrix": penalty_matrix.tolist()
}

with open("romania_cities_with_costs.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Fisierul romania_cities_with_costs.json a fost generat cu succes!")
