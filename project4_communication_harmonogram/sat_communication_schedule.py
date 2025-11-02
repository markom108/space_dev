import json
import requests
import sys
from skyfield.api import load, wgs84, EarthSatellite
import time 
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#-----------------PARAMETRY----------------
SAT_NAME="ISS (ZARYA)"
NR_SATELLITE=25544
SITE_TLE= "https://celestrak.org/NORAD/elements/stations.txt" #AKTUALIZOWANE CO KILKA DNI/H, strona która zawiera TLE twojego oobiektu (kształt orbity, prędkość kątową, pozycję w danym momencie, zmiany w czasie)
GROUND_STATIONS = [
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122, "min_elev": 20},  
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050, "min_elev": 10},
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "min_elev": 15},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "min_elev": 15},
    {"name": "Mauna Kea", "lat": 19.8207, "lon": -155.4681, "min_elev": 10},
    {"name": "Siding Spring", "lat": -31.2733, "lon": 149.0701, "min_elev": 10}
]
REFRESH=5  #co ile GODZIN odświerzamy nasz harmonogram
PREDICT_TIME=10 #na ile GODZIN do przodu przewidujemy okna
STEP=10 #SEC co ile sekund sprawdzać czy już jest w oknie
DEGREE=10#od jakiej wartości stopni nad horyzontem zaczynamy liczyć okno
ONLY_STATION_NIGHT=True #chcę tylko przeloty które, gdy na stacji naziemnej jest ciemno
ONLY_VISIBLE=True #chcę tylko przeloty w których satelita NIE JEST W CIENIU ZIEMII (jest oświetlony przez słońce) 
ts = load.timescale() #tworzy skalę czasu Skyfield, potrzebną do tworzenia momentów astronomicznych
eph = load('de421.bsp')#tzw. efemerydy — plik zawierający bardzo dokładne pozycje ciał niebieskich (Słońca, Ziemi, Księżyca itd.) w czasie    

#-------------------FUNKCJE-----------------------

def load_data(): #OK
    ''' Pobiera TLE dla wszystkich dostępnych satelit(informacje o satelitach i jego orbicie)
        Przeszukuje w celu znalezienia tego satelity którego szukamy.
        Po znalezieniu go zwraca obiekt klasy EarthSatellite w bibliotece Skyfield,
        słóżący do obliczania pozycji w dowolnym momencie'''
    satellites = load.tle_file(SITE_TLE)# wczytuje TLE all sat-> zwraca obiekty Earthorbit
    for sat in satellites:
        if sat.name == SAT_NAME:
            return sat
    sys.exit(f"ERROR: {SAT_NAME} not found in {SITE_TLE}.")

def is_visible(satellite, t):
    '''Sprawdza, czy satelita jest widoczny gołym okiem (czyli oświetlony przez Słońce, nad horyzontem,
    a stacja znajduje się w nocy).

    PRZYDATNE GDY:
    -chcę wiedzieć kiedy można gołym okiem zobaczyć np. ISS z konkretnego punktu na ziemii
    -planuję sfotografować satelitę (np. flare (błysk Iridium))
    - planuję zobaczyć/sfotografować albo przelot przez tarczę Księżyca/Słońca.

    PARAMETRY:
        satellite - obiekt satelity Skyfield (z TLE)
        t - obiekt czasu Skyfield (ts.utc(...))

    ZWRACA:
        TTrue → satelita w świetle Słońca (widoczny np. o zmierzchu)
        False → satelita w cieniu Ziemi (niewidoczny)

    ZASADA DZIAŁANIA:
    Wyobraź sobie prostą linię Słońce-satelita: jeśli ta linia przecian kulę Ziemii, 
    to znaczy, że Ziemia znajduje się między satelitą a Słońcem-> satelita jest w cieniu.
    
    JAK DZIAŁA:
    sun,earth - obiekt reprezentujący tor ruchu Słońca (trajektorię) względem barycentrycznym Układu Słonecznego (SSB), czyli względem środka masy całego układu

    v_sun/v_sat:
        To pozycja Słońca/satelity względem środka Ziemi w danym momencie t.
        Układ współrzędnych: ECI (Earth-Centered Inertial).
        Reprezentuje punkt w przestrzeni, np. (x, y, z) w km.

    vec_sat_sun = v_sat - v_sun
        Wektor od Słońca do satelity.
        To jest nasza linia światła w przestrzeni, w układzie ECI.

    u = - np.dot(v_sun, vec_sat_sun) / dl
        u to parametr(argument/odpowiednik x) w równaniu prostej punkt na prostej
        Gdybyśmy traktowali linię jak funkcję f(x) = v_sun + x * vec_sat_sun, to:
        x = 0 → Słońce
        x = 1 → satelita
        x = u → punkt na linii najbliżej środka Ziemi.
        Matematycznie: szukamy minimum odległości punktu na prostej od (0,0,0) (środek Ziemi).

    closest_point = v_sun + u * vec_sat_sun
        To jest punkt na linii Słońce → satelita, który jest najbliżej środka Ziemi.

    distance = np.linalg.norm(closest_point)
        Odległość tego punktu od środka Ziemi.
        Jeśli distance > R_earth → linia Słońce → satelita nie przecina Ziemi → satelita oświetlony.
    '''
    #--------------------POBIERZ POZYCJĘ SAT I SUN
    sun=eph['sun'] 
    earth=eph['earth']
    R_earth=6371.0  # promień Ziemi w km

    #--------------------WYZNACZ ICH WEKTORY WZGLĘDEM ŚRODKA ZIEMII 
    v_sun=(sun - earth).at(t).position.km 
    v_sat=satellite.at(t).position.km #pozycja satelity w układzie ECI 
    vec_sat_sun = v_sat - v_sun # LINIA ŚWIATŁA
    
    #-------------------ZNALEZIENIE PKT na linii SŁOŃCE-SAT który jest NAJBLIŻEJ ZIEMII 
    dl = np.dot(vec_sat_sun, vec_sat_sun)  # dł wektora^2
    if dl == 0: # dzielenie przez 0
        return False  

    u = - np.dot(v_sun, vec_sat_sun) / dl
    closest_point = v_sun + u * vec_sat_sun
    distance = np.linalg.norm(closest_point)

    # ------------------CZY LINIA PRZECHODZI PRZEZ Ziemię
    return distance > R_earth   # True jeśli widoczny, False jeśli w cieniu

def is_station_in_night(station_topos, t): #OK
    '''Funkcja sprawdza, czy stacja naziemna znajduje się w tzw "nocy cywilnej" 
    (Noc cywilna = Słońce co najmniej threshold_deg stopni poniżej horyzontu (Horyzont to po prostu płaszczyzna styczna do Ziemi w punkcie stacji)
   
    PRZYJMUJE:
    satellite - obiekt Skyfield typu Topos, czyli lokalizacja stacji (szerokość, długość geograficzna, wysokość nad poziomem morza).
    t - obiekt czasu Skyfield (z ts.utc(...)), czyli moment, w którym chcemy sprawdzić, czy jest noc.
    
    ZWRACA:
    True → stacja jest w nocy (Słońce poniżej progu)
    False → stacja jest oświetlona (Słońce nad progiem) 
    
    JAK TO DZIAŁA:
    Skyfield:
        -działa w ECI
        - Wszystkie wektory liczone względem środka Ziemi → łatwiej zrobić obliczenia kosmiczne
    Punkt obserwatora (Topos) - Skyfield wie, gdzie jest stacja na tej kuli.
    Obserwacja Słońca - Skyfield rysuje wyobrażony wektor: od stacji do Słońca. 

    '''
    #----------------USTALAMY WYSOKOŚĆ SŁOŃCA PONIŻEJ HORYZONTU,  którą uznajemy za noc
    threshold_deg=-6

    #----------------OBLICZAMY POZYCJĘ SŁOŃCA (punkt odniesienia: satelita)
    '''
    station (ECI)- wpółrzędne reprezuntujące połozenia stacji w przestrzeni 3D w układzie inercjalnym, którego początkiem jest środek Ziemi (ECI)
    
    station_now(ECI)- używa eph i rotacji Ziemii w czasie do PRZESUNIĘCIA W CZASIE punktu ze station 
    
    sun_pos:  
    1.station_now.observe(eph['sun'])
        Skyfield patrzy ze współrzędnych stacji (station_now w ECI) na Słońce (eph['sun']).
        Oblicza wektor ze stacji do Słońca w układzie ECI, który opisuje kierunek i odległość od stacji do Słońca w km.
        Skyfield opakowuje ten wektor w obiekt klasy Astrometric, który przechowuje zarówno pozycję, jak i kierunek Słońca względem obserwatora.
    2. .apparent()
        Konwertuje pozycję Słońca na położenie pozorne z punktu widzenia obserwatora.
        Uwzględnia efekty fizyczne:
            refrakcję atmosferyczną (załamanie światła w atmosferze),
            aberrację ruchu światła (przesunięcie pozycji z powodu ruchu obserwatora względem źródła światła),
            ewentualnie paralaksę, jeśli obserwator jest na powierzchni Ziemi.
    
    '''
    
    station=eph['earth']+station_topos
    station_now=station.at(t) 
    sun_pos=station_now.observe(eph['sun']).apparent()
    alt, az, dist = sun_pos.altaz()

    return alt.degrees < threshold_deg #jeśli Słońce poniżej threshold -> noc

def compute_windows(satellite, ground_station, hours):#OK
    '''WSTĘP: Satelita krąży wokół Ziemi po orbicie (opisanej przez TLE).
    Na podstawie TLE możemy obliczyć pozycję satelity w dowolnym momencie czasu przyszłego lub przeszłego.
    Ale nie da się policzyć „od razu” całego okna komunikacyjnego
    Dlatego musimy sprawdzić, o której godzinie satelita jest nad horyzontem, a o której znika.
    
    PRZYJMUJE:
    satellite - obiekt satelity z Skyfield (z TLE)
    ground_station - słownik z informacjami o stacji naziemnej zawierający m.in. lat i lon
    hours- ile godzin na przód chcemy obserwować pod kątem okna widoczność
    
    ZWRACA:
    windows - LISTĘ OKIEN WIDOCZNOŚCI w najbliższych hours godzinach dla tej konkretnej stacji
    '''
    windows =[] #lista SŁOWNIKÓW, do której będziemy dodawać momenty, kiedy satelita jest widoczny
    station_topos= wgs84.latlon(ground_station['lat'], ground_station['lon'])#tworzy obiekt Skyfield reprezentujący lokalizację stacji (punkt na Ziemi)
    start = ts.utc(datetime.now(timezone.utc)) 
    end= ts.utc(datetime.now(timezone.utc) + timedelta(hours=hours))
    times, events = satellite.find_events(station_topos, start, end, altitude_degrees=ground_station.get("min_elev", DEGREE))
    '''satellite.find_events:
    OPIS:
    (...).find_events(observer, t0, t1, altitude_degrees=0.0)
    to metoda w Skyfield która:
        - oblicza pozycję satelity względem obserwatora (stacji naziemnej),
        - śledzi jego elewację (wysokość nad horyzontem) w czasie od t0 do t1,
        - automatycznie znajduje momenty kluczowe:
        - kiedy satelita wschodzi nad zadany kąt elewacji (np. 10°),
        - kiedy osiąga kulminację (maksymalną wysokość),
        - kiedy zachodzi poniżej tej elewacji.
        Zamiast ręcznie pętli po sekundach, dostajesz gotowe, bardzo dokładne momenty ↑ (rise), • (culmination) i ↓ (set)
    
    JAK DZIAŁA:
        Skyfield w środku robi bardzo sprytne rzeczy — zamiast brute-force iterować w czasie, stosuje:
        1.Interpolację orbitalną — TLE przekształca w pozycje satelity w układzie ECI (Earth-Centered Inertial).
        2.Transformację do lokalnego układu horyzontalnego obserwatora (AltAz).
        3.Numeryczne wyszukiwanie miejsc zerowych funkcji „elewacja - próg”:
            -szuka, gdzie elewacja przekracza zadaną wartość (np. 10° → wschód),
            - gdzie wraca poniżej (zachód),
            - oraz gdzie elewacja osiąga maksimum (kulminacja).

        4.Wyniki są obliczane z dokładnością do ułamków sekundy.
        Czyli Skyfield nie „symuluje” sekund po sekundzie — on rozwiązuje równanie:
        altitude(t) = altitude_degrees
        przy pomocy metod numerycznych i znajomości wektorów ruchu.

        To dlatego find_events() jest:
        szybki (analizuje np. 24h w sekundę),
        dokładny (dużo bardziej niż własne pętle).

    PRZYJMUJE:
        -obiekt typu Topos,czyli pozycja obserwatora na Ziemi (czyli station_topos = wgs84.latlon(lat, lon))
        -t0, t1: Obiekty czasu Skyfield (Time), określające początek i koniec przedziału analizy
        -altitude_degrees→ Minimalna elewacja (w stopniach), dla której chcesz wyznaczyć moment wejścia i wyjścia.Domyślnie: 0.0 (czyli horyzont).
    ZWRACA: 
    tablicę momentów zdażeń(obiekty Time), 
    tablice events- tablica kodów 0/1/2, odpowiadających zdarzeniom:
        0 = rise (przejście powyżej zadanego kąta), 
        1 = culmination (maks elewacja), 
        2 = set (przejście poniżej kąta np.spadek poniżej 10°)'''
    start=None
    for moment, event in zip(times, events):
        if event==0: #  ↑
            start=moment.utc_datetime()
        elif event==2 and start:#już ↑ było oraz ↓ -> zamykamy okno
            end=moment.utc_datetime()
            if (ONLY_VISIBLE and is_visible(satellite,moment)) or (ONLY_STATION_NIGHT and is_station_in_night(station_topos, moment)) or (ONLY_VISIBLE==False and ONLY_STATION_NIGHT==False):
                windows.append({'start': start, 'end': end, "duration": end-start})    
            start=None
        print(moment.utc_datetime(), ['↑', '•', '↓'][event])
    return windows

#--------------------WIZUALIZACJE------------
def visualization(windows):
    print(f"\n=== Nowy harmonogram dla {SAT_NAME} ===")
    windows.sort(key=lambda x: x[1]['start'])
    for station, w in windows:
        start = w['start'].strftime("%H:%M:%S")
        end = w['end'].strftime("%H:%M:%S")
        bar = "-" * int((w['end'] - w['start']).total_seconds() // 60)
        print(f"{start} -> {end} | {station['name']:10} | {bar}")

'''def Gantt_chart(windows):
    fig, ax = plt.subplots(figsize=(10,2))
    for station, w in windows:
        start = w['start']
        end = w['end']
        station_name = station['name']
        ax.barh(station_name, (end - start).total_seconds()/3600, left=mdates.date2num(start))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('Czas UTC [h]')    
    plt.show()'''

def Gantt_chart(windows):
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Ciemne tło
    fig.patch.set_facecolor('#2E2E2E')
    ax.set_facecolor('#2E2E2E')
    
    # Lista unikalnych stacji
    station_names = [s['name'] for s, _ in windows]
    station_names = sorted(list(set(station_names)))
    
    #okna do nawiązania łączności
    colors = {'visible_night': '#00FF00',  #satelita widoczny gołym okiem+stacja naziemna jest w nocy
              'night_only': '#FFD700',    #stacja jest w nocy
              'in_shadow': '#A9A9A9'} #nie jest widoczny gołym okiem ale jest widoczny na horyzoncie
    
    for station, w in windows:
        start = w['start']
        end = w['end']
        duration_hours = (end - start).total_seconds()/3600
        
        # Ustal kolor paska
        visible = is_visible(TLE, ts.utc(start)) if ONLY_VISIBLE else True
        night = is_station_in_night(wgs84.latlon(station['lat'], station['lon']), ts.utc(start)) if ONLY_STATION_NIGHT else True
        
        if visible and night:
            color = colors['visible_night']
        elif night:
            color = colors['night_only']
        else:
            color = colors['in_shadow']
        
        ax.barh(station['name'], duration_hours, left=mdates.date2num(start), height=0.4, color=color, edgecolor='black')
    
    ax.xaxis_date()
    
    # Dokładniejsza podziałka czasu
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))   # co 1 godzinę
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15)) # co 15 minut
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Kolor osi i etykiet na biały
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white', which='major')
    ax.tick_params(axis='x', colors='white', which='minor')
    ax.tick_params(axis='y', colors='white')
    
    ax.set_xlabel('Czas UTC [h]')
    ax.set_ylabel('Stacja naziemna')
    ax.set_title(f'Harmonogram przelotów {SAT_NAME}')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


#-----------------------PROGRAM-----------------------
while True:
    #-------------------WCZYTANIE DANYCH TLE----------------
    TLE=load_data()
    WINDOWS=[] #lista par (nazwa stacji, słownik:informacje o oknie)
    print(f"SEARCHING FOR COMMUNICATION WINDOWS FOR {SAT_NAME} at {ts.utc(datetime.now(timezone.utc)).utc_datetime()}")
    for station in GROUND_STATIONS: #przechodzi po kluczach (nazwa stacji)
        windows=compute_windows(TLE, station, PREDICT_TIME)
        print(f"Loading windows for {station['name']} station...")
        for w in windows:
            WINDOWS.append((station,w))
        if len(windows)==0:
            print(f"For {station['name']} there is no communication window from {ts.utc(datetime.now(timezone.utc)).utc_datetime()} up to {PREDICT_TIME} hours")
        print("\n")
    
    visualization(WINDOWS)
    Gantt_chart(WINDOWS)
    print(f"Czekam {REFRESH} hours na ponowne poszukiwanie.\n")
    time.sleep(3600*REFRESH)
    
