import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import albumentations as A

# ---------------- PARAMETRY ----------------
DATA_PATH = "38-Cloud_training/"
RESOLUTION = 256
FILTERS = 32
TEST_SIZE = 0.2
SEED = 42
EPOCHS = 50
BATCH_SIZE = 8

# ---------------- AUGMENTACJE ----------------
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
])

def augment_data(image, mask):
    augmented = augmenter(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# ---------------- DANE ----------------
def load_data():
    dict_red = os.path.join(DATA_PATH, "train_red")
    dict_green = os.path.join(DATA_PATH, "train_green")
    dict_blue = os.path.join(DATA_PATH, "train_blue")
    dict_mask = os.path.join(DATA_PATH, "train_gt")
    
    if not all(os.path.exists(p) for p in [dict_red, dict_green, dict_blue, dict_mask]):
        raise Exception("ERROR: Some data folders do not exist.")

    files_gt = os.listdir(dict_mask)
    PATCHES, MASKS = [], []

    for mask_name in files_gt:
        path_mask = os.path.join(dict_mask, mask_name)
        path_red = path_mask.replace("train_gt/gt_", "train_red/red_")
        path_green = path_mask.replace("train_gt/gt_", "train_green/green_")
        path_blue = path_mask.replace("train_gt/gt_", "train_blue/blue_")

        if not all(os.path.exists(p) for p in [path_red, path_green, path_blue, path_mask]):
            continue

        red = cv2.imread(path_red, cv2.IMREAD_UNCHANGED)
        green = cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
        blue = cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)
        rgb = np.dstack((red, green, blue))
        rgb = cv2.resize(rgb, (RESOLUTION, RESOLUTION))
        rgb = rgb.astype(np.float32)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val

        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)[..., np.newaxis]

        rgb_aug, mask_aug = augment_data(rgb, mask)
        PATCHES.append(rgb_aug)
        MASKS.append(mask_aug)

    PATCHES = np.array(PATCHES, dtype=np.float32)
    MASKS = np.array(MASKS, dtype=np.float32)
    return PATCHES, MASKS

# ---------------- DICE + BCE LOSS ----------------
def bce_dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss

# ---------------- ATTENTION BLOCK ----------------
def attention_block(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(g)
    add = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi = layers.Conv2D(1, 1, strides=1, padding='same')(add)
    psi = layers.Activation('sigmoid')(psi)
    return layers.multiply([x, psi])

# ---------------- CONVOLUTION BLOCK ----------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

# ---------------- BUILD U-NET++ WITH ATTENTION ----------------
def build_unetpp(input_shape=(RESOLUTION, RESOLUTION, 3), filters=FILTERS):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = conv_block(inputs, filters)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, filters*2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, filters*4)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = conv_block(p3, filters*8)
    p4 = layers.MaxPooling2D(2)(c4)

    c5 = conv_block(p4, filters*16)

    # Decoder with Attention
    d1 = layers.UpSampling2D()(c5)
    c4_att = attention_block(c4, d1, filters*8)
    d1 = layers.concatenate([d1, c4_att])
    d1 = conv_block(d1, filters*8)

    d2 = layers.UpSampling2D()(d1)
    c3_att = attention_block(c3, d2, filters*4)
    d2 = layers.concatenate([d2, c3_att])
    d2 = conv_block(d2, filters*4)

    d3 = layers.UpSampling2D()(d2)
    c2_att = attention_block(c2, d3, filters*2)
    d3 = layers.concatenate([d3, c2_att])
    d3 = conv_block(d3, filters*2)

    d4 = layers.UpSampling2D()(d3)
    c1_att = attention_block(c1, d4, filters)
    d4 = layers.concatenate([d4, c1_att])
    d4 = conv_block(d4, filters)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=['accuracy'])
    return model

# ---------------- POST-PROCESSING (tylko morfologia) ----------------
def post_process_mask(mask_prob, rgb_image):
    mask = (mask_prob[...,0] > 0.5).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# ---------------- TRENING ----------------
PATCHES, MASKS = load_data()
if len(PATCHES) == 0:
    raise Exception("❌ Brak danych – sprawdź folder 38-Cloud_training i nazwy plików!")

X_train, X_val, y_train, y_val = train_test_split(PATCHES, MASKS, test_size=TEST_SIZE, random_state=SEED)

model = build_unetpp()
print("Training Attention U-Net++...")
start_time = time.time()
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE)
print(f"Training duration: {time.time() - start_time:.2f} sec")

# ---------------- TEST I WIZUALIZACJA ----------------
def test_model(model, patch_index=0):
    rgb_patch = X_val[patch_index]
    gt_mask = y_val[patch_index]
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
    plt.title("Predicted Mask (Morphology + Attention U-Net++)")
    plt.axis('off')
    plt.show()

test_model(model)
