import cv2 #biblioteka OpenCV, wczytywanie i przetwarzanie obrazów
import numpy as np #wygodna praca z macierzami (obrazy w formie tablic)
import matplotlib.pyplot as plt #wyświetlanie obrazów i wykresów
import os #moduł służący do interakcji z systemem operacyjnym (pliki, foldery, ścieżki)
from sklearn.ensemble import RandomForestClassifier #importujemy klasyfikator czyli model learningowy typu nazwie Random Forest
from sklearn.model_selection import train_test_split #funkcja losowo dzieląca dane na test i training
from sklearn.metrics import accuracy_score #do oceny dokładności modelu (poprawność między 85% a 100% jest okej)
import time #liczenie czasu

TEST_SIZE=0.2#ile procent przykładów idzie na test (reszta danych idzie na trening)
SEED=42

#----------------------ETAP 1:WCZYTAJ DANE
path_red= "38-Cloud_training/train_red/red_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF"
path_green="38-Cloud_training/train_green/green_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF"
path_blue="38-Cloud_training/train_blue/blue_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF"
path_mask="38-Cloud_training/train_gt/gt_patch_6_1_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.TIF"

#sprawdzenie poprawności ścieżki
if not os.path.exists(path_red) or not os.path.exists(path_green) or not os.path.exists(path_blue) or not os.path.exists(path_mask):
    print("ERROR 1: layer path does not exist.")

red_layer=cv2.imread(path_red, cv2.IMREAD_UNCHANGED) #macierz 2D liczb przedstawiająca natężenie koloru
green_layer=cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
blue_layer=cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)

rgb=np.dstack((red_layer,green_layer,blue_layer))#łączymy 3 macierze w jedną macierz, która opisuje każdy piksel wartością (r,g,b), czyli wartość koloru dla każdego piksela
mask=cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)# wartości od 0-255 oznaczające tylko chmury

plt.figure(figsize=(10,5))#ustawiam rozmiar okna
plt.subplot(1,3,1)#dzielę okno na 1 wiersz i 2 kolumny, oraz aktywuję pierwszy obszar
plt.imshow(rgb / np.max(rgb))  # normalizacja, moje obrazy są 16-bitowe, a imshow oczekuje 0-1 lub 8-bitowe: skalujemy w odniesieniu do najjaśniejszego piksela na mapie, żeby zachować proporcje, oraz czytelność obrazu
plt.title("Obraz RGB")
plt.axis("off") #wyłącz oś wokół wykresu

plt.subplot(1,3,2)#aktywuj drugi obszar
plt.imshow(mask / np.max(mask), cmap="gray")#interpretuj liczby jako odcień szarości
plt.title("Maska chmur")
plt.axis("off")

#----------------------ETAP 2: Przygotowanie danych oraz tworzenie prostego modelu uczenia maszynowego, który przypisuje etykiety do danych (dane=[R,G,B], etykieta=1-należy do chmury, 0- nie należy do chmury) => W czasie treningu model uczy się zależności między wartościami [R, G, B] a etykietą 0/1.
#rgb=[[[1,22,31],[...]]] (3D) => cechy=[[1,22,31], [...],[...]] (2D)

cechy=rgb.reshape(-1,3) #lista cech wszystkich pikseli=[R,G,B] (3D=>2D), liczba kolumn=3(R,G,B), liczba wierszy=-1(samo policz: (H*W*3)/3 )
mask=(mask>0).astype(np.uint8)#porównaj z 0 i zwróć tablicę bool i skonwertuj z boola na int 0 lub 1, żeby nie zostawiać wartości logicznych które mogą powodować błędy
etykiety=mask.reshape(-1) #lista 1D etykiet: 0-nie jest chmurą, 1-jest chmurą

#--------------Tworzę ML
model=RandomForestClassifier(n_estimators=100, random_state=SEED)#Tworzenie: liczba drzew decyzyjnych, ustalam "ziarno losowości"- to liczba, która ustala punkt startowy dla generatora liczb pseudolosowych w modelu. Dzięki temu wszystkie losowe decyzje (np. wybór próbek i cech w Random Forest) stają się powtarzalne – ten sam random_state daje zawsze te same wyniki treningu, ale nie zmienia jakości modelu
cechy_train,cechy_test,etykiety_train,etykiety_test=train_test_split(
        cechy,etykiety, test_size=TEST_SIZE, random_state=SEED
    )#DZIELIMY DANE na testowe i treningowe
print("Model in training...")
start_time=time.time()
model.fit(cechy_train, etykiety_train)#trenuj model
end_time=time.time()
print(f"Training duration: ({end_time-start_time:.2f}) sec" )

print("Testing model...")
etykiety_predicted=model.predict(cechy_test)#egzamin: sprawdzamy jakie odpowiedzi dla testów da nasz model
accuracy=accuracy_score(etykiety_test, etykiety_predicted)#porównuje poprawne i przewidywane odpowiedzi i ocenia skuteczność modelu
print("Accuracy on test set: ", accuracy)

plt.show()