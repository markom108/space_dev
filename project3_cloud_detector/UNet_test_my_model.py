import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ---------------- PARAMETRY ----------------
RESOLUTION = 256
DATA_PATH = "38-Cloud_training/"
BLUE_NAME = "blue_patch_85_5_by_5_LC08_L1TP_002054_20160520_20170324_01_T1.TIF"#sprawdzane zdjęcie
MODEL_PATH = "UNet_cloud_detector_NN_model.h5"
model = load_model("UNet_cloud_detector_NN_model.h5", compile=False)# wczytanie modelu

# ---------------- PROGRAM ----------------

def post_process_mask(mask_predict, rgb_image):
    mask = (mask_predict[...,0] > 0.5).astype(np.uint8) #zamienia na 0/1
    kernel = np.ones((3,3), np.uint8)# zwykła macierz jedynek 3x3: Służy do patrzenia na sąsiadujące piksele przy czyszczeniu maski
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)#"zamknięcie" dziór: Erozja → Dylatacja| cv2.MORPH_CLOSE= Erozja(ŚCISKA- Piksel pozostaje 1 tylko jeśli wszystkie sąsiednie piksele w kernelu są 1=wypełnij dzióry) -> Dylatacja(ROZSZERZE-Każdy piksel obiektu „sprawdza” swoje sąsiedztwo w kernelu i jeśli któryś sąsiad jest 1 → staje się 1=łączy małe dzióry w obiekcie)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)#Dylatacja → Erozja
    return mask #Masz teraz czystą, binarną maskę gotową do wizualizacji albo dalszych obliczeń

def test_model_by_name(model, blue_name):
    #---------------------PRZYGOTOWANIE DANYCH------------------------
    # Tworzymy nazwy innych kanałów i maski
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
        rgb_patch /= max_val  # normalizacja do 0-1

    # Wczytanie maski GT 
    gt_mask = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
    gt_mask = cv2.resize(gt_mask, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 0).astype(np.uint8)[..., np.newaxis]  # konwersja na 0/1

    #---------------------PREDYKCJA MASKI---------------------------------
    input_tensor = np.expand_dims(rgb_patch, axis=0)  # dodaj wymiar batch
    start_time=time.time()
    pred_mask_prob = model.predict(input_tensor)[0]  # usuwa wymiar batch
    elapsed_time=time.time()-start_time

    #maska po morfologii
    pred_mask_pp = post_process_mask(pred_mask_prob, rgb_patch)

    # ----------------WIZUALIZACJA-----------------------------
# Wizualizacja wyników
    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    plt.imshow(rgb_patch)
    plt.title("Input RGB")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(gt_mask[...,0], cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(pred_mask_prob[...,0], cmap='gray')
    plt.title("Predicted Mask (raw)")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(pred_mask_pp, cmap='gray')
    plt.title("Predicted Mask (Morphology)")
    plt.axis('off')

    plt.show()

# ---------------- WCZYTANIE MODELU ----------------
model = load_model(MODEL_PATH, compile=False)

# ---------------- TEST ----------------
test_model_by_name(model, BLUE_NAME)