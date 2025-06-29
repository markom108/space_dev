'''
CEL:
	1. Symulacja zbierania danych (sensor data simulation)
	2. Pakowanie danych w formacie JSON (packet formatting)
    3.(Później) Symulacja transmisji przez TCP lub zapis danych jakby były wysyłane drogą radiową
    4. ładnie graficznie to obrobić: kolorki na errorach, wykresy itp

SPIS PROGRAMU:
alerts()- funkcja zwracająca błąd, jęśli dane pokazane przez sensory przekroczyłu ustawiony limit
generate_fake_telemetry()- Symulujesz dane z sensorów (temperature, voltage) 

teraz testujemy jeszcze to    
'''

import time
import random
import json # do zmieniania formatu na JSON

'''KONSOLA DLA SPECJALLISTY:'''
    #acceptable temperature parameters
MIN_TEMP=20
MAX_TEMP=39
    #acceptable voltage parameters
MIN_VOLT=6.5
MAX_VOLT=8
#INNE:
TEST=True
ID_START=0
SAT_ID="SAT-001" #satellite id
RECORD=True #czy chcemy zapisywać wyniki do pliku
TRYB_ZAP="w" # w- nadpisuje, jeśli chcemy dodawać to w trybie a 
SAVE_INTERVAL=5
'''KONIEC KONSOLI'''


def alerts(volt, temp):
    if MIN_VOLT>volt or volt>MAX_VOLT:
        return "voltage",volt
    if MIN_TEMP>temp or temp>MAX_TEMP:
        return "temperature",temp
    return 0,0

def generate_fake_telemetry(packet_id): #simulation of data from sensors
    voltage=round(random.uniform(5.5,8.5),2) # between 6.5- 8.5 V
    temperature=round(random.uniform(20.0,40.0),2) #between 20 and 40
    
    ERROR, value= alerts(voltage, temperature)
   
    timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    results ={
        "satellite_id":SAT_ID,
        "packet_id": packet_id,
        "timestamp": timestamp, #czas wysłania
        "Error": (ERROR,value),
        "voltage": voltage,
        "temperature": temperature
        }
    return results 

def main():
    packet_id=ID_START
    if RECORD:
        logfile=open("telemetry_log.jsonl",TRYB_ZAP)# otwiera plik telemetry_log w trybie zapisu, logfile = obiekt pliku który może wykonywać operacje zapisu
    
    last_flush=time.time()#czas ostatniego zapisu(aktualny czas w sekundach)
    while TEST:
        data=generate_fake_telemetry(packet_id)
        if RECORD:
            json_packet= json.dumps(data) #zmiana data na format json
            logfile.write(json_packet+"\n")
            now=time.time()
            if now-last_flush>=SAVE_INTERVAL:
                logfile.flush() #dane będą zapisywane do telemetry_log.jsonl "na żywo" podczas działania programu
                last_flush=now
        print("Telemetry data: ", data)
        time.sleep(1)
        packet_id+=1


    logfile.close()#ważne żeby zamknąć na końcu, żeby wszystko poprawnie zapisane
main()

