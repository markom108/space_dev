import skyfield.api as sf #do pobierania TLE z popularnych źródeł jak nasa.gov celestrak.org
import datetime as dt
import time
from collections import deque

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

'''---------------KONSOLA UŻYTKOWNIKA-------------------------------------------------'''
NAME ="ISS (ZARYA)"#name of searched object
SITE= "https://celestrak.org/NORAD/elements/stations.txt" #strona która zawiera TLE twojego oobiektu (kształt orbity, prędkość kątową, pozycję w danym momencie, zmiany w czasie)

REFRESH=5 #ilość sekund pomiędzy odświerzeniami
FUTURE_TRAJ=60 #odstęp (sekundy) miezy ponownym pobraniem TLE

#--------------MAP parametrs--------------------
TRAJ_TIME_PERIOD=30 #(sec)ile czasu do przodu obliczyć do wyświetlenia przyszłych pozycji w jednym czasie na mapie (90)
TRAJ_STEP=3 #(sec) ile czasu odstępu między obliczanymi pozycjami (60) 
LAST_TRACKER=130 #ile ostatnich punktów z trasy satelity pokazywać

satellite=None
count=0
konwerter=sf.load.timescale() #tworzy obiekt typu Timescale, konwerter na kosmiczny

'''-----------PRZEWIDYWANIE PRZYSZŁYCH TRAJEKTORII------------------------------------------------------------------------'''
def generate_trajectory(satellite):#przewidywana w danym momencie PRZYSZŁA trajektoria, na podstawie aktualnych parametrów
    POSITIONS={}
    now=dt.datetime.now(dt.timezone.utc)#aktualny w UTC

    for added in range(0, TRAJ_TIME_PERIOD+1,TRAJ_STEP):
        czas_kosmiczny=konwerter.utc(now+dt.timedelta(seconds=added))#umośliwia dodanie konkretnej ilości np. sek. do czasu w formacie UTC
        pos_obj=satellite.at(czas_kosmiczny) #obiekt dający pozycję satelity
        position=pos_obj.subpoint()#metoda pozwalająca na obliczenie dokładnego miejsca nad ziemią
        POSITIONS[now+dt.timedelta(seconds=added)]= (position.latitude.degrees, position.longitude.degrees, position.elevation.km) #zapisanie do słownika pozycji
    return POSITIONS

'''--------------RYSOWANIE STATYCZNEJ MAPY----------------------------------------------'''
def create_page():
    okno=plt.figure(figsize=(16,8), facecolor='black')
    #MAPA:
    wykres_mapa=okno.add_axes([0.05, 0.1, 0.72, 0.85], projection=ccrs.Robinson())
    draw_map(wykres_mapa) #rysowanie RAZ (niezmienne)

    #LEGENDA
    wykres_legenda=okno.add_axes([0.75, 0.6, 0.18, 0.3])
    draw_legend(wykres_legenda) #rysowanie RAZ (niezmienne)

    #GLOBUS
    wykres_globe=okno.add_axes([0.8, 0.05, 0.18, 0.3], projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    draw_globe(wykres_globe)
    plt.suptitle(f"{NAME} - Ground Track Visualization", color='white', fontsize=18, y=0.96)
    return  wykres_mapa, wykres_globe

def draw_legend(ax):
    ax.axis('off')#wyłącza rysowanie osi
    legend_text=("Legend:\n"
          "-- last prediction"
          "__ test")
    ax.text(0,1,legend_text, color='white', fontsize=12,fontfamily='monospace',verticalalignment='top',bbox=dict(facecolor='black',alpha=0.6, edgecolor='lime', pad=8))

def draw_map(ax):#Rysuje JEDEN RAZ dużą mapę Robinsona (eliptyczny rzut) na podanej osi
    #----background----
    ax.set_global()#cała mapa świata
    ax.set_facecolor('black') #kolor tła osi

    #-----map style----
    ax.add_feature(cfeature.LAND, facecolor='green')#dodaj warstwę lądu do mapy
    ax.add_feature(cfeature.OCEAN,facecolor='navy')#dodaje warstwę oceanu
    ax.add_feature(cfeature.BORDERS, linestyle=':')#dodaje granice
    #ax.add_feature(cfeature.RIVERS, linewidth=0.2,color='lightblue')
    #ax.add_feature(cfeature.LAKES, linewidth=0.2,color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='white')

    #-----siatka------
    siatka=ax.gridlines(draw_labels=True,color='white',alpha=0.5, linestyle='--', linewidth=0.5)#etykiety z wartościami geograficznymi, kolor, przeźroczystość lini, rodzaj linii, szerokość linii
    siatka.top_labels=siatka.right_labels=False #Wyłącza (powtarzające się) etykiety na górnej i prawej krawędzi mapy.
    siatka.xlabel_style=siatka.ylabel_style={'color': 'white', 'fontsize': 8}#styl siatki

def draw_globe(ax):#Rysuje globus Orthographic z ramką.
    ax.set_global()#Ustawia zasięg mapy tak, aby obejmowała cały glob, czyli pełną kulę ziemską.
    ax.set_facecolor('black')
    
    #---- edycja mapy------
    ax.add_feature(cfeature.LAND, facecolor='lime')
    ax.add_feature(cfeature.OCEAN, facecolor='navy')
    ax.add_feature(cfeature.COASTLINE, color='white', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    #------siatka-------
    ax.gridlines(draw_labels=False, linewidth=0.5)
    #-----ramka------
    rect = Rectangle((0, 0), 1, 1, linewidth=0.2, edgecolor='gray', facecolor='none',
                     transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)#dodanie do wykresu obiektu ramki
    
'''--------------LIVE AKTUALIZACJE-------------------------'''
live_pointer_map=None #uchwyt= referencja do artysty (lista zawierająca punkty np. jeden punkt live ISS)
FUTURE_line_map=None
LAST_line_map=None
previous = deque(maxlen=LAST_TRACKER)  # automatycznie usuwa najstarsze punkty
trail_line=None

def update_tracking_map(ax, now, future_positions):
    global trail_line,live_pointer_map, FUTURE_line_map, LAST_line_map #bo modyfikujemy w całym programie, nie tylko w funkcji
    lat=now.latitude.degrees
    lon=now.longitude.degrees
    previous.append((lon,lat))#dodaje z prawej poprzednie współrzędne
    
    #------LIVE POINT---------    (red)
    if live_pointer_map: #jeśli poprzednio był jakiś punkt
        live_pointer_map.remove()#usuwanie poprzedniej kropki
    live_pointer_map=ax.plot(lon,lat,marker='o', color='red', markersize=6,transform=ccrs.Geodetic(), label='Aktualna pozycja na mapie')[0]#RYSYJ PUNKT =... ,podaję współrzędne w długości/szerokości geogr
    
    #-----LIVE LINE------ (red, -)
    if len(previous)>1:
        lons_hist, lats_hist=zip(*previous)
        if trail_line:#usuwamy starą linię śladu
            trail_line.remove()
        trail_line=ax.plot(lons_hist, lats_hist, color='red', linewidth=2, transform=ccrs.Geodetic())[0]


    #------ FUTURE------- (lime, -)
    if future_positions :
        lats = [pos[0] for pos in future_positions.values()]#lista zawierająca szer geogr. w kolejności czasowej
        lons = [pos[1] for pos in future_positions.values()]#lista zawierająca dł geogr. w kolejności czasowej

        if LAST_line_map: #jeśli była już przewidziana jakaś trasa
            LAST_line_map.remove()

        if FUTURE_line_map:
            LAST_line_map=FUTURE_line_map
            LAST_line_map.set_color('orange')
            LAST_line_map.set_linestyle('--')
            LAST_line_map.set_markersize(1)

        FUTURE_line_map=ax.plot(lons, lats, color='lime', linestyle='-', linewidth=2,transform=ccrs.Geodetic(), label='Aktualna przyszła trajektoria')[0]# rysujemy linię trajektorii na mapie

def update_tracking_globe(ax, now, positions):
    

    return

'''----------------------KOD---------------------------------------'''
plt.ion()#interaktywne odświerzanie mapy(żeby nie trzeba było zamykać okna i otwierać nowego)
chart_map, chart_globe=create_page()
print(f"Loading data of {NAME}...")

while True:
    #------------pobieranie i parsowanie pliku-------------------
    satellites = sf.load.tle_file( SITE ) #lista obiektów EarthSatellite ze sprasowanymi danymi TLE (połaczeone 3 linijki stąd .name .line1 .line2) 
    for i in satellites:
        if i.name==NAME:
            satellite=i
    if satellite==None:
        print(f"ERROR: {NAME} does not exist. Please check its name or try changing TLE site.")
        continue

    #---------FUTURE TRACKER-----------------------
    future_positions={}
    if count%FUTURE_TRAJ==0: # czy teraz wyliczamy kolejną przyszłą trajektorię
        future_positions=generate_trajectory(satellite)

    #----------LIVE TRACKING----------------------------
    now=dt.datetime.now(dt.timezone.utc) #aktualny czas w formacie UTC
    now=konwerter.utc(now)#konwertuje aktualny czas z UTC na kosmiczny
    position=satellite.at(now)#zwraca objekt typu Geocentric -(można z tego wziąć, prędkość, współrzędne, punkt nad ziemią lub obserwowanie coś z tej pozycji)

    subpoint = position.subpoint()#punkt na powierzchni ziemi bezpośrednio POD SATELITĄ  
    #print(f" Longtitude: {subpoint.longitude.degrees:.2f}°\n", f"Latitude: {subpoint.latitude.degrees:.2f}°\n", f"Height: {subpoint.elevation.km:.2f} km\n")#Długość, szerokość, wysokość geograficzna
    
    #----------MAP-------------------
    update_tracking_map(chart_map,subpoint,future_positions)
    #update_tracking_globe(chart_globe, subpoint, future_positions)

    plt.pause(0.01)#teraz rysuj to co wcześniej zapisane i poczekaj
    count+=REFRESH
    time.sleep(REFRESH)


