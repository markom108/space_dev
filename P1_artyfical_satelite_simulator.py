''' GENERATOR DANYCH
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
import socket #pozwala na połączeni a sieciowe
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
RECORD=False #czy chcemy zapisywać wyniki do pliku
TRYB_ZAP="w" # w- nadpisuje, jeśli chcemy dodawać to w trybie a 
SAVE_INTERVAL=5

#łączenie z satelitą
SERVER=True
if SERVER: 
    client_socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)# (określenie jakiego formatu adresu IP będziemy używać,określenie typu komunikacji))
    #dane serwera(gdzie ma się łączyć) - te same co w ground station
    host='127.0.0.1' #lokalny host, bo wszystko na tym samym komputerze
    port=5000


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

    if SERVER:

        client_socket.connect((host,port))#tworzenie połączenia TCP z moim serwerem (najpierw chcę połączyć się z serwerem, zanim zaczne wysyłać dane)
    
    last_flush=time.time()#czas ostatniego zapisu(aktualny czas w sekundach)
    
    try: # kod, który może rzucić wyjątek (BŁĄD-> PRZERYWA, probi finally i wyrzuca błąd)
        while TEST:
            data=generate_fake_telemetry(packet_id)
            json_packet= json.dumps(data) #zmiana data na format json
            if RECORD:#czy chcesz zapisywać do pliku
                logfile.write(json_packet+"\n")
                now=time.time()
                if now-last_flush>=SAVE_INTERVAL:
                    logfile.flush() #dane będą zapisywane do telemetry_log.jsonl "na żywo" podczas działania programu (trochę spowalnia program)
                    last_flush=now
            
            if SERVER: #czy chcemy przesyłać na server
                client_socket.send(json_packet.encode())#wysyłanie pakietu

            #print("Telemetry data: ", data)
            time.sleep(1)
            packet_id+=1
    finally: # ten kod wykona się zawsze, nawet jeśli wyjątek wystąpi
        if SERVER:
            client_socket.close()
        if RECORD:
            logfile.close()#ważne żeby zamknąć na końcu, żeby wszystko poprawnie zapisane
main()

