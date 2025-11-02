''' Symulowanie działania danych:
Program, symuluje generowanie danych z czujników satelity i przesyłanie ich na server/ zapisywanie do pliku w czasie rzeczywistym.

'''
import socket #pozwala na połączeni a sieciowe
import time
import random
import json # do zmieniania formatu na JSON
from colorama import init,Fore,Style #umożliwia kolorowanie tesktu w terminalu
init(autoreset=True) # po każdym kolorowym tekście, automatycznie przywróć domyślny kolor teminala

from user_console import MIN_TEMP, MAX_TEMP, MIN_VOLT,MAX_VOLT, ID_START, SAT_ID, RECORD, SAVE_INTERVAL, TRYB_ZAP, SERVER, host, port

    
#łączenie z satelitą
if SERVER: 
    client_socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)# (określenie jakiego formatu adresu IP będziemy używać,określenie typu komunikacji))


def alerts(volt, temp):
    if MIN_VOLT>volt or volt>MAX_VOLT:
        return "volt",volt
    if MIN_TEMP>temp or temp>MAX_TEMP:
        return "temp",temp
    return 0,0

def generate_fake_telemetry(packet_id): #simulation of data from sensors
    voltage=round(random.uniform(5.5,8.5),2) # between 6.5- 8.5 V
    temperature=round(random.uniform(20.0,40.0),2) #between 20 and 40
    
    ERROR, value= alerts(voltage, temperature)
    if ERROR== 0:
        status="OK"
    else:
        status=f"ERROR: {ERROR} {value}"
   
    timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    results ={
        "satellite_id":SAT_ID,
        "packet_id": packet_id,
        "timestamp": timestamp, #czas wysłania
        "status": status,
        "voltage": voltage,
        "temperature": temperature
        }
    return results 

def main():
    if RECORD:
        logfile=open("telemetry_log.jsonl",TRYB_ZAP)# otwiera plik telemetry_log w trybie zapisu, logfile = obiekt pliku który może wykonywać operacje zapisu

    if SERVER:
        client_socket.connect((host,port))#tworzenie połączenia TCP z moim serwerem (najpierw chcę połączyć się z serwerem, zanim zaczne wysyłać dane)
    
    packet_id=ID_START
    last_flush=time.time()#czas ostatniego zapisu(aktualny czas w sekundach)
    
    try: # kod, który może rzucić wyjątek (BŁĄD-> PRZERYWA, probi finally i wyrzuca błąd)
        while True:
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

            #wypisywanie z kolorowaniem statusunie m
            status=data["status"]
            if status=="OK":
                status_colored=Fore.GREEN + status #kolor + co koloruję + wróć do domyślnych ustawień kolorów i stylów
            else:
                status_colored=Fore.RED + status
            
            for key, value in data.items():
                if key=="status":
                    print(f"{key}: {status_colored}")
                else:
                    print(f"{key}: {value}")
            print()
            time.sleep(1)
            packet_id+=1
    
    finally: # ten kod wykona się zawsze, nawet jeśli wyjątek wystąpi
        #ZAMYKANIE
        if SERVER:
            client_socket.close()
        if RECORD:
            logfile.close()#ważne żeby zamknąć na końcu, żeby wszystko poprawnie zapisane


main()

