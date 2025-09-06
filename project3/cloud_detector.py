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
DATA_PATH="38-Cloud_training/"

dict_red=f"{DATA_PATH}train_red/"
dict_green=f"{DATA_PATH}train_green/"
dict_blue=f"{DATA_PATH}train_blue/"
dict_mask=f"{DATA_PATH}train_gt/"
if not os.path.exists(dict_red) or not os.path.exists(dict_green) or not os.path.exists(dict_blue) or not os.path.exists(dict_mask):
    print("ERROR 1: dict path does not exist.")

files_gt=os.listdir(dict_mask)#lista wszystkich plików w fold
FEATURES=[] #(R,G,B) cechy wszystkich fragmentów, wejściowe dla ML
LABELS=[] #(0/1) etykiety wszystkich fragmentów, odpowiedzi

#------------------ETAP 1: PRZYGOTOWANIE DANYCH
for mask_name in files_gt:
    path_mask=os.path.join(DATA_PATH,"train_gt", mask_name)
    path_red= path_mask.replace("train_gt/gt_", "train_red/red_")
    path_green= path_mask.replace("train_gt/gt_", "train_green/green_")
    path_blue= path_mask.replace("train_gt/gt_", "train_blue/blue_")
    if not os.path.exists(path_red) or not os.path.exists(path_green) or not os.path.exists(path_blue) or not os.path.exists(path_mask):
        print("ERROR 2: layer path does not exist.")

    #----------------------WCZYTANIE KOLORÓW: macierz 2D liczb przedstawiających natężenie koloru
    red_layer=cv2.imread(path_red, cv2.IMREAD_UNCHANGED)
    green_layer=cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
    blue_layer=cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)

    rgb=np.dstack((red_layer,green_layer,blue_layer))#łączymy w POTRÓJNĄ TABLICĘ: jedna macierz 2D, każdy punkt ma przypisaną wartość [R,G,B] opisującą kolor jednego piksela: rgb=[[[1,22,31],[...]]]
    mask=cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)# MACIERZ 2D: wartości od 0-255 oznaczające tylko chmury: [[1,22,31], [...],[...]]

    #---------------------- PRZYGOTOWANIE DANYCH:  macierze 2D => LISTY  1D 
    cechy_patch=rgb.reshape(-1,3) #LISTA: liczba wierszy=-1(samo policz: (H*W*3)/3 ),liczba kolumn=3(R,G,B)
    mask=(mask>0).astype(np.uint8) #zamień wartości 0-255 na bool i skonwertuj T/F na 0/1
    etykiety_patch=mask.reshape(-1) #lista 1D etykiet: 0-nie jest chmurą, 1-jest chmurą

    #----------------------DODANIE DO ZBIORU
    FEATURES.append(cechy_patch)
    LABELS.append(etykiety_patch) 


#--------------ETAP 2: Tworzę ML i testuję na wszystkich patchach
model=RandomForestClassifier(n_estimators=100, random_state=SEED)#Tworzenie: liczba drzew decyzyjnych, ustalam "ziarno losowości"- to liczba, która ustala punkt startowy dla generatora liczb pseudolosowych w modelu. Dzięki temu wszystkie losowe decyzje (np. wybór próbek i cech w Random Forest) stają się powtarzalne – ten sam random_state daje zawsze te same wyniki treningu, ale nie zmienia jakości modelu
#   DZIELIMY DANE na testowe i treningowe
#FEATURES i LABELS to listy tablic ==> trzeba połączyć w jedną tablicę
cechy_train,cechy_test,etykiety_train,etykiety_test=train_test_split(
        FEATURES,LABELS, test_size=TEST_SIZE, random_state=SEED
    )

print("Model in training...")
start_time=time.time()
model.fit(cechy_train, etykiety_train)#trenuj model
end_time=time.time()
print(f"Training duration: ({end_time-start_time:.2f}) sec" )

print("Testing model...")
etykiety_predicted=model.predict(cechy_test)#egzamin: sprawdzamy jakie odpowiedzi dla testów da nasz model
accuracy=accuracy_score(etykiety_test, etykiety_predicted)#porównuje poprawne i przewidywane odpowiedzi i ocenia skuteczność modelu
print("Accuracy on test set: ", accuracy)

'''

plt.figure(figsize=(10,5))#ustawiam rozmiar okna
plt.subplot(1,3,1)#dzielę okno na 1 wiersz i 2 kolumny, oraz aktywuję pierwszy obszar
plt.imshow(rgb / np.max(rgb))  # normalizacja, moje obrazy są 16-bitowe, a imshow oczekuje 0-1 lub 8-bitowe: skalujemy w odniesieniu do najjaśniejszego piksela na mapie, żeby zachować proporcje, oraz czytelność obrazu
plt.title("Obraz RGB")
plt.axis("off") #wyłącz oś wokół wykresu

plt.subplot(1,3,2)#aktywuj drugi obszar
plt.imshow(mask / np.max(mask), cmap="gray")#interpretuj liczby jako odcień szarości
plt.title("Maska chmur")
plt.axis("off")

plt.show()'''