import cv2 #biblioteka OpenCV, wczytywanie i przetwarzanie obrazów
import numpy as np #wygodna praca z macierzami (obrazy w formie tablic)
import matplotlib.pyplot as plt #wyświetlanie obrazów i wykresów
import os #moduł służący do interakcji z systemem operacyjnym (pliki, foldery, ścieżki)
from sklearn.model_selection import train_test_split #funkcja losowo dzieląca dane na test i training
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import time #liczenie czasu
import sys
import random
import albumentations as A #do augmentacji

'''WSTĘP: Sieci Neuronowe (NN) 
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

# ---------------- PARAMETRY ----------------
DATA_PATH="38-Cloud_training/"
RESOLUTION=256 #(rozdzielczość) na jaką rozdzielczość zdjęcia chcemy zamienić w U-NET
FILTERS=32 #liczba filtrów dla U-Net
TEST_SIZE=0.2  #ile procent przykładów idzie na test (reszta danych idzie na trening)
SEED=42
EPOCHS=50
BATCH_SIZE=8
TEST_GROUP_SIZE=2000
SAVE=True #do you want to save current NN model?

# ---------------- AUGMENTACJE ---------------- 
'''Nie zawsze mamy wystarczającą ilość danych, stąd bieżemy zdjęcia które mamy dane
i modyfikujemy je: obracamy, odbijamy itp. tak żeby postały "nowe" lekko inne zdjęcia,
na których NN może się uczyć
#augmenter = obiekt klasy (Compose), który w sobie przechowuje listę możliwych transformacji, po użyciu na obrazie,
obraz zostaje transformowany, używając wylosowanych cech zapisanych właśnie w augmenterze.
'''

augmenter = A.Compose([ 
    A.HorizontalFlip(p=0.5),         # losowe odbicie w poziomie, prawdopodobieństwo zastosowania: 50%
    A.VerticalFlip(p=0.5),           # losowe odbicie w pionie
    A.RandomRotate90(p=0.5),         # losowy obrót o 90°
    A.RandomBrightnessContrast(p=0.3), # zmiana jasności i kontrastu
    A.GaussNoise(p=0.2),             # dodanie szumu gaussowskiego
    A.MotionBlur(p=0.2),             # rozmycie ruchowe
])

def augmenter_data(image,mask): #funkcja która TRANSFORMUJE obraz
    augmented=augmenter(image=image,mask=mask)#słownik ze zmodyfikowanym w TEN SAM SPOSÓB obrazem i maską
    return augmented['image'], augmented['mask']


#-----------------DANE------------------

def load_data():
    global DATA_PATH, TEST_GROUP_SIZE
    dict_red = os.path.join(DATA_PATH, "train_red")
    dict_green = os.path.join(DATA_PATH, "train_green")
    dict_blue = os.path.join(DATA_PATH, "train_blue")
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
        print("Load patch nr: ",count+1, " ")
        
        path_mask=os.path.join(DATA_PATH,"train_gt", mask_name)
        path_red= path_mask.replace("train_gt/gt_", "train_red/red_")
        path_green= path_mask.replace("train_gt/gt_", "train_green/green_")
        path_blue= path_mask.replace("train_gt/gt_", "train_blue/blue_")
        if not all(os.path.exists(p) for p in [path_red, path_green, path_blue, path_mask]):
            print(f"ERROR 2: path for layer of {mask_name} does not exist.")
            continue #dont stop the program, just ignore

        #------------WCZYTANIE KOLORÓW i KONWERSJA na format U-net: macierz 2D liczb przedstawiających natężenie koloru (U-NET OCZEKUJE DANYCH WEJŚCIOWYCH w formacie tensora 4D numpy)
        red_layer=cv2.imread(path_red, cv2.IMREAD_UNCHANGED)#MACIERZ 2D: korzystanie z OpenCV do odczytu obrazka z pliku(co, wczytaj nie zmieniając)
        green_layer=cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
        blue_layer=cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)
        rgb=np.dstack((red_layer,green_layer,blue_layer))#łączymy w TENSOR 3D (lista 3D): jedna macierz 2D, każdy punkt ma przypisaną wartość [R,G,B] opisującą kolor jednego piksela: rgb=[[[1,22,31],[...]]]
        rgb=cv2.resize(rgb,(RESOLUTION,RESOLUTION))
        #NORMALIZACJA
        rgb = rgb.astype(np.float32)# Zmieniasz typ danych obrazu na float32 (liczby zmiennoprzecinkowe), bo NN lepiej na nich działają
        max_val = np.max(rgb)#największą wartość w obrazie.
        if max_val > 0:#zabezpiecza przed dzieleniem przez 0 (czarny obraz)
            rgb /= max_val #skalowane wszystkich pikseli do zakresu 0–1

        mask=cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)# MACIERZ 2D: wartości od 0-255 oznaczające tylko chmury: [[1,22,31], [...],[...]]
        mask = cv2.resize(mask, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)#(H,W,1) KONWERSJA NA FORMAT DANYCH WEJŚCIA DLA U-NET | metoda interpolacji "najbliższym sąsiadem" - normalnie przy np. powiększaniu obrazu wartości nowych pikseli będą średnią sąsiadów, ale jak że to jest maska my chcemy  tylko 0/1 dlatego używamy tej opcji
        mask = (mask > 0).astype(np.uint8)[..., np.newaxis]#zamień wartości 0-255 na bool i skonwertuj T/F na 0/1 + dodanie kanału, tak żeby format dla wejścia się zgadzał

        rgb_aug,mask_aug=augmenter_data(rgb,mask)
        PATCHES.append(rgb_aug)
        MASKS.append(mask_aug)

    # zamiana list na tensory numpy 4D: Aby wrzucić je do sieci, najlepiej mieć jedną tablicę 4D: (liczba_patchy, wysokość, szerokość, kanały)
    PATCHES=np.array(PATCHES, dtype=np.float32)#NN oczekuje danych w formacie float32
    MASKS=np.array(MASKS, dtype=np.float32)

    return PATCHES, MASKS

# ---------------- DICE + BCE LOSS ----------------
'''MIERZENIE POPRAWNOŚCI MODELU:
Model potrzebuje feedbacku, po obliczeniu przewidywanej maski potrzebuje informacji jak bardzo sie mylił,
żeby mógł się poprawić. Można sprawdzać na różne sposoby, ale dobrym połączeniem jest BCE+Dice

    * BCE (Binary Crossentropy) 
Lokalny błąd piksel po pikselu
1.Oblicza BCE ze wzoru dla każdego pixela
2.Sumujemy/bierzemy średnią BCE po wszystkich pikselach → wynik 

    * Dice Loss
Patrzy globalnie, na całą maskę jako całość.
Obliczamy nakładanie się (intersection) przewidywanej maski z prawdziwą:
1. A=Mnożymy odpowiadające sobie piksele y_true * y_pred i sumujemy → im większa suma, tym lepiej
2. B=Sumujemy sumy wartości maski i predykcji
3. Korzystamy ze wzoru na Dice_loss=1-(2*A/B)

Sam BCE → sieć może nauczyć się ignorować małe obiekty
BCE + Dice → sieć widzi zarówno piksele, jak i całą strukturę obiektu

    Jak loss wpływa na wagi i biasy?
Sieć przewiduje maskę 
Obliczamy loss (BCE + Dice) → liczba mówiąca „jak bardzo przewidywanie różni się od prawdy”
TensorFlow oblicza gradienty loss względem każdej wagi i biasu
Sieć aktualizuje wagi i biasy, żeby loss zmniejszyć w kolejnej iteracji
'''

def bce_dice_loss(odp, pred):
    smooth=1.0 #liczba dodawana do licznika i mianownika, żeby uniknąć dzielenia przez zero(np. gdy maska cała czarna)
    vector_odp=tf.keras.backend.flatten(odp)#Zamieniamy macierz w jednowymiarowe wektory
    vector_pred=tf.keras.backend.flatten(pred)
    #BCE
    bce = tf.keras.losses.BinaryCrossentropy()(odp, pred)
    #Dice
    intersection=tf.reduce_sum(vector_odp*vector_pred)
    dice_loss=1-(2*intersection+smooth)/(tf.reduce_sum(vector_odp)+tf.reduce_sum(vector_pred)+smooth)
    return bce+dice_loss

#----------------------BUDOWA U-Net:------------------
def build_U_Net():
    input_shape=(RESOLUTION,RESOLUTION,3)
    filters=FILTERS
    inputs= layers.Input(input_shape)#OPIS danych WEJŚCIA NN: to wywołanie tworzy obiekt typu KerasTensor, który opisuje kształt danych wejściowych.
    
    #----------------ENCODER------------
    def conv_block(inputs,filters): #(Conv2D+BatchNorm)x2
        x=layers.Conv2D(filters,3, activation='relu', padding='same')(inputs)#warstwa konwolucyjna w 2D (HxW)[liczba filtrów w jednej konwolusji | rozmiar przesuwanego okna np. 3x3 | Aktywacja funkcji ReLU (Rectified Linear Unit): Usuwa ujemne wartości(ujemne=oznacza brak cechy lub negatywna infomracja) |   dodawanie „otoczki” z zer pikseli wokół obrazu
        x=layers.BatchNormalization()(x) #Jeśli wartości tych danych są bardzo duże/małe, sieć uczy się wolno ==> ta funkcja „przeskaluje” wartości wyjściowe każdej warstwy tak, żeby były w miarę stabilne i podobne w każdej warstwie
        x=layers.Conv2D(filters,3, activation='relu', padding='same')(x)
        x=layers.BatchNormalization()(x)
        return x
    
    c1=conv_block(inputs, filters) #convert: warstwa 1
    p1=layers.MaxPooling2D(2)(c1) #pool: warstwa 1  :[dzieli mapę na bloki 2×2 i wybiera maksimum w każdym bloku](to co poolingujemy)

    c2=conv_block(p1, filters*2)
    p2=layers.MaxPooling2D(2)(c2)

    c3=conv_block(p2, filters*4)
    p3=layers.MaxPooling2D(2)(c3)

    c4=conv_block(p3, filters*8)
    p4=layers.MaxPooling2D(2)(c4)
    #2.BOTTLENECK (najgłębsza warstwa-ostatnia konwolucja)
    c5=conv_block(p4, filters*16)

    #3.DECODER
    def decoder_block(last,skip_conn,filters): #Upsample+concatenate: ostatnia warstwa, odpowiedni blok,liczba filtrów
        x=layers.UpSampling2D()(last)
        x=layers.concatenate([x,skip_conn])
        x=conv_block(x,filters)
        return x

    d1=decoder_block(c5,c4,filters*8)
    d2=decoder_block(d1,c3,filters*4)
    d3=decoder_block(d2,c2,filters*2)
    d4=decoder_block(d3,c1,filters)
    
    outputs=layers.Conv2D(1,1, activation='sigmoid')(d4)#WARSTWA WYJŚCIOWA: ostatnia filtracja: 1 filtr(czy należy do chmury czy nie), 1x1 rozmiar filtra, funkcja aktywacji daje wartość 0/1: ta warstwa bierze ostatnią mapę cech z decodera i przekształca ją w maskę binarną, gdzie każdy piksel ma wartość 0–1.
    # opis działania modelu
    model=models.Model(inputs, outputs)#obiekt modelu Keras, łączy wszystkie warstwy [dane wejściowe, dane wyjściowe]
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=['accuracy'])#jak trenować, jak oseniać poprawność: (algorytm optymalizacji wag sieci podczas uczenia | funkcja straty: mierzy jak bardzo przewidywana maska różni się od prawdziwej maski|dodatkowa metryka do monitorowania skuteczności w trakcie treningu.)
    #model.summary()# WYPISUJE W KONSOLI: szybkie podsumowanie całej architektury Twojej sieci neuronowej. Pokazuje Ci wszystkie warstwy i ich parametry w czytelnej formie
    return model


#--------------------TRENOWANIE---------------------
PATCHES, MASKS=load_data()
if len(PATCHES) == 0:
    sys.exit("Error 3: No files, please check your folder and names of files. ")

cechy_train,cechy_test,etykiety_train,etykiety_test=train_test_split(PATCHES,MASKS, test_size=TEST_SIZE, random_state=SEED)
model=build_U_Net()

print("Training U-Net...")
start=time.time()
model.fit(cechy_train, etykiety_train, validation_data=(cechy_test, etykiety_test), 
        epochs=EPOCHS, batch_size=BATCH_SIZE)#epochs→ sieć przechodzi przez wszystkie dane treningowe 5 razy|batch_size → nie przekazujemy całego zbioru na raz, tylko partiami po 8 obrazów. Po każdej partii sieć uczy się| history → obiekt, w którym zapisują się wszystkie informacje o treningu, np. strata (loss) i dokładność (accuracy) dla treningu i walidacji w każdej epoce. Możemy je potem wykorzystać do rysowania wykresów uczenia.
print(f"Training duration: ({time.time()-start:.2f}) sec" )

if SAVE==True:
    print("Saving...")
    model.save("UNet_cloud_detector_NN_model.h5")#ZAPISANIE MODELU


# ---------------- POST-PROCESSING (tylko morfologia) ----------------
'''CZYSZCZENIE, UZUPEŁNIANIE:
U-Net przewiduje maskę dla każdego piksela, np. które piksele należą do chmury, a które do tła.
Jednak to wyjście często jest „miękkie” i lekko „szumne”: ma małe dziury w chmurach albo pojedyncze białe punkty w tle.

To "czyszczenie"/naprawę nazywamy MORFOLOGIĄ
Morfologia obrazu to zestaw operacji matematycznych, które analizują i przetwarzają kształty w obrazie binarnym (0 = tło, 1 = obiekt).
Używa się jej głównie w segmentacji i oczyszczaniu masek. Dzięki temu maska jest:
-Binarna (0 lub 1)
-Spójna (bez dziur w obiektach, bez pojedynczych szumów)

PO CO?
-Aby maska była bardziej realistyczna
-Aby uniknąć problemów w dalszym przetwarzaniu (np. liczenie chmur, obliczanie powierzchni)
'''

def post_process_mask(mask_predict, rgb_image):
    mask = (mask_predict[...,0] > 0.5).astype(np.uint8) #zamienia na 0/1
    kernel = np.ones((3,3), np.uint8)# zwykła macierz jedynek 3x3: Służy do patrzenia na sąsiadujące piksele przy czyszczeniu maski
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)#"zamknięcie" dziór: Erozja → Dylatacja| cv2.MORPH_CLOSE= Erozja(ŚCISKA- Piksel pozostaje 1 tylko jeśli wszystkie sąsiednie piksele w kernelu są 1=wypełnij dzióry) -> Dylatacja(ROZSZERZE-Każdy piksel obiektu „sprawdza” swoje sąsiedztwo w kernelu i jeśli któryś sąsiad jest 1 → staje się 1=łączy małe dzióry w obiekcie)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)#Dylatacja → Erozja
    return mask #Masz teraz czystą, binarną maskę gotową do wizualizacji albo dalszych obliczeń

# ---------------- TEST I WIZUALIZACJA ----------------
def test_model(model, patch_index=0):
    rgb_patch = PATCHES[patch_index]
    gt_mask = MASKS[patch_index]
    input_tensor = np.expand_dims(rgb_patch, axis=0)
    pred_mask_prob = model.predict(input_tensor)[0]
    pred_mask_pp = post_process_mask(pred_mask_prob, rgb_patch)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(rgb_patch)
    plt.title("Input RGB")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(gt_mask[...,0], cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_mask_pp, cmap='gray')
    plt.title("Predicted Mask (Morphology only)")
    plt.axis('off')
    plt.show()


'''def test_model_after_files_name(model):
    print("Visual test loading... ")
    # Nazwa niebieskiego patcha
    blue_name = "blue_patch_85_5_by_5_LC08_L1TP_002054_20160520_20170324_01_T1.TIF"
    red_name = blue_name.replace("blue_", "red_")
    green_name = blue_name.replace("blue_", "green_")
    gt_name = blue_name.replace("blue_", "gt_")
    # Ścieżki do plików
    path_red = f"{DATA_PATH}train_red/{red_name}"
    path_green = f"{DATA_PATH}train_green/{green_name}"
    path_blue = f"{DATA_PATH}train_blue/{blue_name}"
    path_gt = f"{DATA_PATH}train_gt/{gt_name}"  
    
    # Wczytanie obrazów
    red_layer = cv2.imread(path_red, cv2.IMREAD_UNCHANGED)
    green_layer = cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
    blue_layer = cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)

    # Połączenie w RGB i dopasowanie rozdzielczości
    rgb_patch = np.dstack((red_layer, green_layer, blue_layer))
    rgb_patch = cv2.resize(rgb_patch, (RESOLUTION, RESOLUTION))
    rgb_patch = rgb_patch.astype(np.float32)
    max_val = np.max(rgb_patch)
    if max_val > 0:
        rgb_patch = rgb_patch / max_val # normalizacja do 0-1

    # Wczytanie maski GT 
    answer = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
    answer = cv2.resize(answer, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)
    answer = (answer > 0).astype(np.uint8)   # konwersja na 0/1
    answer = answer[..., np.newaxis]

    # Predykcja maski przez model
    input_tensor = np.expand_dims(rgb_patch, axis=0)  # dodaj wymiar batch
    pred_mask = model.predict(input_tensor)[0]  # usuwa wymiar batch -> (H, W, 1)
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

    # Wizualizacja wyników
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(rgb_patch)
    plt.title("Input RGB")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(answer, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_mask_binary[:,:,0], cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()'''

test_model(model)