import json
import time
from datetime import datetime

#-------------------USER INTERFACE--------------
KEY_WORD = "id"
BATCH_SIZE = 2
SAFE_BATTERY = 0.8
CRITICAL_THRESHOLD = 0.05
REFRESH = 1
MANEUVERS = 5
SAFETY_MARGIN = 0.1

#----------------FUNKCJE---------------
def simulate_docking(sat):
    """Symulacja dokowania i ładowania satelity"""
    leaving_dock = False

    if sat["distance_to_station"] > 0:
        step = min(sat.get("speed_km_per_sec", 20) * REFRESH, sat["distance_to_station"])
        sat["distance_to_station"] -= step
        sat["energy"] -= step * sat.get("energy_per_km", 0.1)
        sat["status"] = "moving to docking"
    else:
        # satelita dociera do stacji
        sat["status"] = "charging"
        charge_step = sat.get("charge_rate", 5)
        target_energy = sat["capacity"] * SAFE_BATTERY
        sat["energy"] = min(sat["energy"] + charge_step, target_energy)

        # jeśli energia >=80%, oznaczamy jako fully charged i opuszcza dock
        if sat["energy"] / sat["capacity"] >= SAFE_BATTERY:
            sat["status"] = "fully charged"
            leaving_dock = True

    progress = int((sat['energy'] / sat['capacity']) * 100)
    progress = min(progress, 100)
    print(f"{sat['id']} [{'#'*progress}{'.'*(100-progress)}] {sat['energy']:.1f}% | {sat['status']}")
    if leaving_dock:
        print(f"{sat['id']} is leaving the dock!")
    return sat

def print_global_snapshot(step, sat_states):
    print(f"\n=== GLOBAL SNAPSHOT #{step} ===")
    print(f"{'ID':<8} {'Energy(%)':<10} {'Dist(km)':<10} {'Status'}")
    for sat in sat_states.values():
        print(f"{sat['id']:<8} {sat['energy']:<10.1f} {sat['distance_to_station']:<10} {sat['status']}")
    print("-" * 65)

#-----------------LOAD DATA--------------------
with open("satellites_static.json") as f:
    static_data = json.load(f)

with open("satellites_dynamic.json") as f:
    live_data = json.load(f)

#--------------------------MAIN--------------------
sat_states = {}   # id -> aktualny stan
queue_dict = {}   # id -> satelita w kolejce do ładowania

for i in range(0, len(live_data), BATCH_SIZE):
    batch = live_data[i:i + BATCH_SIZE]
    records = []

    # Aktualizacja stanów satelitów
    for data in batch:
        key = data[KEY_WORD]
        if key not in static_data:
            continue

        if key in sat_states and sat_states[key]["status"] in ("charging", "moving to docking"):
            # jeśli satelita już ładuje się lub dociera, ignorujemy nowe dane
            records.append(sat_states[key])
            continue

        # nowy stan lub aktualizacja
        if key not in sat_states:
            sat_states[key] = {**static_data[key], **data}
        else:
            sat_states[key].update(data)

        energy_ratio = sat_states[key]['energy'] / sat_states[key]['capacity']

        if energy_ratio <= CRITICAL_THRESHOLD:
            sat_states[key]['status'] = "critical"
        elif energy_ratio >= SAFE_BATTERY:
            sat_states[key]['status'] = "fully charged"
        else:
            sat_states[key]['status'] = "waiting for docking"

        records.append(sat_states[key])

    # Budowanie kolejki ładowania
    for sat in records:
        if sat["status"] in ("waiting for docking", "critical"):
            queue_dict[sat["id"]] = sat

    # Obsługa kolejki - tylko jeden satelita może ładować się naraz, chyba że jest krytyczny
    if queue_dict:
        critical_sats = [s for s in queue_dict.values() if s["status"] == "critical"]

        if critical_sats:
            sat_to_charge = min(critical_sats, key=lambda x: x["energy"])
        else:
            sat_to_charge = list(queue_dict.values())[0]

        s = simulate_docking(sat_to_charge)
        if s["status"] not in ("fully charged", "critical"):
            queue_dict[s["id"]] = s
        else:
            queue_dict.pop(s["id"], None)

        # reszta satelitów czeka
        for key, sat in queue_dict.items():
            if sat["id"] != s["id"] and sat["status"] not in ("fully charged", "charging", "critical"):
                sat["status"] = "waiting for docking"
                print(f"{sat['id']} | waiting for docking (energy={sat['energy']:.1f}%)")

    print_global_snapshot(i // BATCH_SIZE + 1, sat_states)
    time.sleep(REFRESH)
