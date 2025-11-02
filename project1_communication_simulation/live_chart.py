''' WYKRES NA ZYWO
Kod otwierający w nowym oknie wykres utworzony z danych przesłanych przez symulowanego satelitę.
'''
import time
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from user_console import MIN_TEMP, MAX_TEMP, MIN_VOLT, MAX_VOLT

NR_RECORDS=30 #ile maksymalnie chcemy pokazywać rekordów w jednym czasie na jednym wykresie
REFRESH_INTERVAL=0.5

def chart():
    plt.ion()#włącza tryb interaktywny - wykres można aktualizowac dynamicznie, bez zatrzymywania programy
    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(1,1,1)
    blink=True

    while True:
        packet_ids=[]
        voltages=[]
        temperatures=[]
        file=open("telemetry_log.jsonl", "r")#w trybie odczytu
        lines=file.readlines()#?
        recent_telemetry=lines[-NR_RECORDS:]#ostatnie x linijek
        xs=[]
        ys=[]
        for line in recent_telemetry:#pobieranie danych 
            data=json.loads(line) #data z jednej z ostatnich linijek
            packet_ids.append(data["packet_id"])
            voltages.append(data["voltage"])
            temperatures.append(data["temperature"])
            if MIN_VOLT>data["voltage"] or data["voltage"]>MAX_VOLT:
                xs.append(data["packet_id"])
                ys.append(data["voltage"])
            if MIN_TEMP>data["temperature"] or data["temperature"]>MAX_TEMP:
                xs.append(data["packet_id"])
                ys.append(data["temperature"])

        
        #RYSOWANIE WYKRESU
        ax.clear()
        ax.set_facecolor('black')
        ax.plot(packet_ids, voltages, label='Voltage', color='cyan')
        ax.plot(packet_ids, temperatures, label='Temperature', color='magenta', linestyle='--')
        
        if xs:
            #xs, ys = zip(*alarm_points)  # rozpakowanie na dwie listy
            ax.scatter(xs, ys, color='red', s=50, zorder=5)  # s=50 to rozmiar kropki, zorder=5 żeby na wierzchu


        # delikatne linie min/max dla napięcia
        ax.axhline(y=MIN_VOLT, color='cyan', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=MAX_VOLT, color='cyan', linestyle='--', linewidth=1, alpha=0.3)

        # delikatne linie min/max dla temperatury
        ax.axhline(y=MIN_TEMP, color='magenta', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=MAX_TEMP, color='magenta', linestyle='--', linewidth=1, alpha=0.3)
        
        ax.set_title('Telemetry Data Live Chart', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Packet ID', color='white', fontsize=12)
        ax.set_ylabel('Value', color='white', fontsize=12)
        
        ax.tick_params(colors='white')
        ax.grid(color='gray', linestyle='--', alpha=0.3)
        
        legend = ax.legend()
        legend.get_frame().set_alpha(0.3)
        for text in legend.get_texts():
            text.set_color('white')

        plt.pause(REFRESH_INTERVAL)#odświerzanie wykresu

        file.close()

chart()