'''KLIENT:'''
    #acceptable temperature parameters
MIN_TEMP=20
MAX_TEMP=39
    #acceptable voltage parameters
MIN_VOLT=6.5
MAX_VOLT=8

ID_START=0
SAT_ID="SAT-001" #satellite id

#zapisywanie do pliku
RECORD=True #czy chcemy zapisywać wyniki do pliku
TRYB_ZAP="w" # w- nadpisuje, jeśli chcemy dodawać to w trybie a 
SAVE_INTERVAL=5

#łączenie z satelitą
SERVER=False

'''ŁĄCZENIE SERVER-KLIENT'''
#dane serwera(gdzie ma się łączyć) - te same co w ground station
host='127.0.0.1' #lokalny host, bo wszystko na tym samym komputerze
port=5000

'''SERWER:'''
MAX_CONECTIONS=1
MAX_RECEIVE=1024 #ile maksymalnie bajtów chcesz odebrać za jednym razem



'''CHART:'''
NR_RECORDS=30 #ile maksymalnie chcemy pokazywać rekordów w jednym czasie na jednym wykresie
REFRESH_INTERVAL=0.5