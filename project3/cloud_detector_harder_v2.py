import cv2 #biblioteka OpenCV, wczytywanie i przetwarzanie obrazów
import numpy as np #wygodna praca z macierzami (obrazy w formie tablic)
import matplotlib.pyplot as plt #wyświetlanie obrazów i wykresów
import os #moduł służący do interakcji z systemem operacyjnym (pliki, foldery, ścieżki)
from sklearn.ensemble import RandomForestClassifier #importujemy klasyfikator czyli model learningowy typu nazwie Random Forest
from sklearn.model_selection import train_test_split #funkcja losowo dzieląca dane na test i training
from sklearn.metrics import accuracy_score #do oceny dokładności modelu (poprawność między 85% a 100% jest okej)
import time #liczenie czasu
import sys
import random

TEST_SIZE=0.2#ile procent przykładów idzie na test (reszta danych idzie na trening)
SEED=42
DATA_PATH="38-Cloud_training/"
TEST_GROUP_SIZE=100
RESOLUTION=256 #(rozdzielczość) na jaką rozdzielczość zdjęcia chcemy zamienić w U-NET

'''Machine Learning (ML):
    Dziedzina sztucznej inteligencji, tworzenie algorytmów, które uczą się na danych i potrafią przewidywać/podejmować pewne decyzje.
PRZYKŁAD ALGORYTMU ML: 
    Random Forest- drzewa decyzyjne'''
def RandomForest():
    dict_red=f"{DATA_PATH}train_red/"
    dict_green=f"{DATA_PATH}train_green/"
    dict_blue=f"{DATA_PATH}train_blue/"
    dict_mask=f"{DATA_PATH}train_gt/"
    if not os.path.exists(dict_red) or not os.path.exists(dict_green) or not os.path.exists(dict_blue) or not os.path.exists(dict_mask):
        sys.exit("ERROR 1: dict path does not exist.")

    files_gt=os.listdir(dict_mask)#lista wszystkich plików w fold
    FEATURES=[] #(R,G,B) cechy wszystkich fragmentów, wejściowe dla ML
    LABELS=[] #(0/1) etykiety wszystkich fragmentów, odpowiedzi

    #------------------ETAP 1: PRZYGOTOWANIE DANYCH
    TEST_GROUP_SIZE=min(TEST_GROUP_SIZE, len(files_gt))
    random_patches=random.sample(files_gt,TEST_GROUP_SIZE)#wylasuj od razu ileś patchy, żeby nie powtórzyć

    for count in range(TEST_GROUP_SIZE):
        mask_name=random_patches[count] #weź randomowy element z listy
        print("Load patch nr: ",count, " ", mask_name)

        path_mask=os.path.join(DATA_PATH,"train_gt", mask_name)
        path_red= path_mask.replace("train_gt/gt_", "train_red/red_")
        path_green= path_mask.replace("train_gt/gt_", "train_green/green_")
        path_blue= path_mask.replace("train_gt/gt_", "train_blue/blue_")
        if not os.path.exists(path_red) or not os.path.exists(path_green) or not os.path.exists(path_blue) or not os.path.exists(path_mask):
            sys.exit("ERROR 2: layer path does not exist.")

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
    cechy_patch=np.vstack(FEATURES)#vertical stack = sklej po wierszach
    etykiety_patch=np.hstack(LABELS)#horizontal stack = sklej po elementach wektorów 1D
    cechy_train,cechy_test,etykiety_train,etykiety_test=train_test_split(
            cechy_patch,etykiety_patch, test_size=TEST_SIZE, random_state=SEED
        )

    print("Model in training...")
    start_time=time.time()
    model.fit(cechy_train, etykiety_train)#trenuj model
    end_time=time.time()
    print(f"Training duration: ({end_time-start_time:.2f}) sec" )

    print("Testing model...")
    etykiety_predicted=model.predict(cechy_test)#egzamin: sprawdzamy jakie odpowiedzi dla testów da nasz model
    accuracy=accuracy_score(etykiety_test, etykiety_predicted)#porównuje poprawne i przewidywane odpowiedzi i ocenia skuteczność modelu
    print(f"Accuracy on test set: {accuracy*100:.6f} %")

'''Sieci Neuronowe (NN) 
Specjalny rodzaj algorytmów w ML (podkategoria ML), który potrafi sie uczyć złożonych zależności w danych.
Składa się z warst neuronów, potrafi samodzielnie wydobywać cechy, wymaga dużo danych i mocy obliczeniowej.
    RODZAJ SIECI NAURONOWYC (np.):
-Convolutional Neural Network (CNN):
Głównie do analizy obrazów, wiele warstw które uczą się cech obrazy
Zamiast patrzeć na pojedyncze piksele,  patrzy na małe fragmenty obrazu (np. 3x3 piksele) i uczy się rozpoznawać wzory.
Dzięki temu sieć potrafi wykrywać krawędzie, kolory, tekstury, a w końcu chmury.
-U-Net:
Rodzaj sieci neuronowej (wariant CNN),stworzony do segmentacji obrazów
U-Net robi tzw. konwolucje → przesuwa małe okienko (np. 3×3) po całym obrazie i uczy się wzorców lokalnych: kształt chmury,tekstura ziemi,przejścia jasności.

    CNN vs U-Net:
CNN klasyczny – np. rozpoznaje, że na obrazie jest pies.
U-Net – mówi dokładnie, które piksele należą do psa, czyli segmentuje obiekt w obrazie.
'''
def CNN():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    dict_red=f"{DATA_PATH}train_red/"
    dict_green=f"{DATA_PATH}train_green/"
    dict_blue=f"{DATA_PATH}train_blue/"
    dict_mask=f"{DATA_PATH}train_gt/"
    if not os.path.exists(dict_red) or not os.path.exists(dict_green) or not os.path.exists(dict_blue) or not os.path.exists(dict_mask):
        sys.exit("ERROR 1: dict path does not exist.")

    files_gt=os.listdir(dict_mask)#lista wszystkich plików w fold
    TEST_GROUP_SIZE=min(TEST_GROUP_SIZE, len(files_gt))
    random_patches=random.sample(files_gt,TEST_GROUP_SIZE)#wylasuj od razu ileś patchy, żeby nie powtórzyć
    PATCHES=[]
    MASKS=[]

    for count in range(TEST_GROUP_SIZE):
        mask_name=random_patches[count] #weź randomowy element z listy(NAZWA PLIKU)
        print("Load patch nr: ",count, " ", mask_name)
        path_mask=os.path.join(DATA_PATH,"train_gt", mask_name)
        path_red= path_mask.replace("train_gt/gt_", "train_red/red_")
        path_green= path_mask.replace("train_gt/gt_", "train_green/green_")
        path_blue= path_mask.replace("train_gt/gt_", "train_blue/blue_")
        if not os.path.exists(path_red) or not os.path.exists(path_green) or not os.path.exists(path_blue) or not os.path.exists(path_mask):
            sys.exit("ERROR 2: layer path does not exist.")

        #------------WCZYTANIE KOLORÓW i KONWERSJA na format U-net: macierz 2D liczb przedstawiających natężenie koloru (U-NET OCZEKUJE DANYCH WEJŚCIOWYCH w formacie tensora 4D numpy)
        red_layer=cv2.imread(path_red, cv2.IMREAD_UNCHANGED)#MACIERZ 2D: korzystanie z OpenCV do odczytu obrazka z pliku(co, wczytaj nie zmieniając)
        green_layer=cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
        blue_layer=cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)
        rgb=np.dstack((red_layer,green_layer,blue_layer))#łączymy w TENSOR 3D (lista 3D): jedna macierz 2D, każdy punkt ma przypisaną wartość [R,G,B] opisującą kolor jednego piksela: rgb=[[[1,22,31],[...]]]
        rgb=cv2.resize(rgb,(RESOLUTION,RESOLUTION))

        mask=cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)# MACIERZ 2D: wartości od 0-255 oznaczające tylko chmury: [[1,22,31], [...],[...]]
        mask=np.dstack(mask)#(H,W,1) KONWERSJA NA FORMAT DANYCH WEJŚCIA DLA U-NET: U-net działą na tablicachy Numpy
        mask=cv2.resize(mask,(RESOLUTION,RESOLUTION))

        #----------------------NORMALIZACJA (ujednolicenie danych wejściowych)
        rgb = rgb / np.max(rgb)  #PRZEJŚCIE DO PRZEDZIAŁU od 0 do 1: Surowe obrazy .TIFF są zazawyczaj 16 bitowe (wartości pikseli to np. od 0 do 65535), a sieci neuronowe lepiej uczą się, gdy dane wejściowe są znormalizowane – czyli w jakimś ujednoliconym zakresie (np. 0–1
        mask=(mask>0).astype(np.uint8) #zamień wartości 0-255 na bool i skonwertuj T/F na 0/1
        
        PATCHES.append(rgb)
        MASKS.append(mask)

    PATCHES=np.array(PATCHES) #Aby wrzucić je do sieci, najlepiej mieć jedną tablicę 4D: (liczba_patchy, wysokość, szerokość, kanały)
    MASKS=np.array(MASKS)
    #---------------PODZIAŁ DANYCH
    cechy_train,cechy_test,etykiety_train,etykiety_test=train_test_split(
            PATCHES,MASKS, test_size=TEST_SIZE, random_state=SEED
        )
    




CNN()