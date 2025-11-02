import json
import time

#-------------------USER INTERFACE--------------
KEY_WORD="id" #po tym słowie wyszukujemy satelitę
THRESHOLD=20 #próg(%) od którego powinien pojawić się alert
BATCH_SIZE=2 #ilość satelitów obserwowanych
SAFE_BATTERY=0.8 #do ilu procent ładujemy
PREDICT_STEPS = 5       # liczba kroków do przodu do przewidywania awarii
REFRESH=1 #co ile odświerzamy w sekundach
MANEUVERS=5 #koszt energetyczny manewrów
SAFETY_MARGIN=0.1 # %
IDLE = "idle"
#----------------FUNKCJE---------------
def predict_failure(sat):
    '''Przewiduje, czy  i w ilu najbliższych "krokach" poziom energii satelity może spaść poniżej krytycznego poziomu energii.'''
    energy=sat["energy"]
    distance = sat["distance_to_station"]
    consumption=sat.get("power_consumption",1) #zużycie utrzymania satelity (nie wliczamy ruchu) per step
    km_per_step=sat.get("speed_km_per_sec", 20)*REFRESH #szacowana dłogość którą pokona
    for step in range(1,PREDICT_STEPS+1):
        safe_threshold=distance*sat.get("energy_per_km", 0.1)+MANEUVERS+SAFETY_MARGIN*sat["capacity"]
        energy-=km_per_step*sat.get("energy_per_km",0.1)
        energy-=consumption
        distance-=km_per_step
        if energy<=safe_threshold:
            return True,step
    return False, PREDICT_STEPS+2

def check_energy(records):
    '''Sprawdza poziom energii satelitów z predykcją awarii. Zwraca listę krotek (priorytet, dane) satelitów wymagających alertu'''
    alerts=[]
    for record in records:
        energy_to_dock=record["distance_to_station"]*record.get("energy_per_km", 0.1)+MANEUVERS
        safe_threshold=energy_to_dock +SAFETY_MARGIN*record["capacity"]
        if record["energy"] <=safe_threshold:# alert jeśli obecnie poniżej safe threshold
            alerts.append((1,record))#(priority,record)
        else:
            result,steps=predict_failure(record)
            if result:         # przewidujemy awarię
                alerts.append((1+steps,record)) #(priority,record)
    return alerts

def generate_alerts(alerts):
    '''Tworzy NIEPOSORTOWANE komunikaty alertów dla satelitów.'''
    messages=[]
    for priority,sat in alerts:
        msg = (f"ALERT: {sat['id']} energy={sat['energy']:.1f}% | "
               f"status={sat['status']} | distance={sat['distance_to_station']} km")
        messages.append(msg)
    return messages

#dodać animacje do ładowania
def simulate_docking(sat):
    """Symulacja zbliżania, dokowania i ładowania w tle"""
    if sat["distance_to_station"] > 0:
        step = min(sat.get("speed_km_per_sec", 20) * REFRESH, sat["distance_to_station"])
        sat["distance_to_station"] -= step
        sat["energy"] -= step * sat.get("energy_per_km", 0.1)
        sat["status"] = "moving to docking"
    else:
        sat["status"] = "docking/charging"
        charge_step = sat.get("charge_rate", 5)
        target_energy = sat["capacity"] * SAFE_BATTERY
        sat["energy"] = min(sat["energy"] + charge_step, target_energy)
        if sat["energy"] >= target_energy:
            sat["status"] = "charged"
    progress = int((sat['energy']/sat['capacity'])*100)
    print(f"{sat['id']} [{'#'*progress}{'.'*(100-progress)}] {sat['energy']:.1f}%")
    return sat

def print_snapshot(step, records):
    print(f"\n=== SNAPSHOT #{step} ===")
    print(f"{'ID':<6} {'Energy(%)':<10} {'Dist(km)':<10} {'Status':<15}")
    for sat in records:
        print(f"{sat['id']:<6} {sat['energy']:<10.1f} {sat['distance_to_station']:<10} {sat['status']:<15}")

#-----------------LOAD DATA--------------------
with open("satellites_static.json") as f:
    static_data =json.load(f) #lista słowników z danymi

with open("satellites_dynamic.json") as f:
    live_data = json.load(f) #lista słowników z danymi w aktualnym momencie(oraz 5 rekordów do tyłu)

#--------------------------MAIN--------------------
queue_dict = {}  # sat_id -> (priority, sat)
for i in range(0, len(live_data), BATCH_SIZE):
    batch = live_data[i:i+BATCH_SIZE]  # bierze kolejne x satelitów
    records=[]

    for data in batch:
        key=data[KEY_WORD]
        temp={**static_data[key], **data}#słownik ze wszystkimi danymi (key_val2 nadpisuje key_val2)
        if "status" not in temp:
            temp["status"] = "idle" #none
        records.append(temp)
    #---------------ALERTS---------------------
    alerts=check_energy(records)#czy energia spadła poniżej progu -> lista alertów
    messages=generate_alerts(alerts)
    for msg in messages:
        print(msg)

    #--------------ADDING TO THE QUEUE------------
    for priority, sat in alerts:
        key=sat[KEY_WORD]
        if key not in queue_dict or priority < queue_dict[key][0]:
            queue_dict[key] = (priority, sat)


    #-------------PROCESSING-----------
    queue_list=sorted(queue_dict.values(), key=lambda x: x[0])
    new_queue_dict={}
    for priority, sat in queue_list:
        if sat["status"] == "charged":
            continue
        s = simulate_docking(sat)
        if s["status"] != "charged":
            new_queue_dict[sat["id"]] = (priority, s)
        print(f"{sat['id']} | status={s['status']} | energy={s['energy']:.1f}% | distance={s['distance_to_station']} km")

    queue_dict = new_queue_dict

    print_snapshot(i // BATCH_SIZE + 1, records)
    print("-" * 65)
    time.sleep(REFRESH)
