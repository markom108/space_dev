import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import time
import sys
import random
import albumentations as A

# ---------------- PARAMETERS ----------------
DATA_PATH = "38-Cloud_training/"
RESOLUTION = 256
FILTERS = 32
TEST_SIZE = 0.2
SEED = 42
EPOCHS = 50
BATCH_SIZE = 8
TEST_GROUP_SIZE = 2000
SAVE = True

# ---------------- AUGMENTATION ----------------
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
])

def augmenter_data(image, mask):
    """Apply augmentation to image and mask using albumentations."""
    augmented = augmenter(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# ---------------- DATA LOADING ----------------
def load_data():
    """Load RGB image layers and corresponding masks, apply augmentation, normalize, and prepare tensors."""
    global DATA_PATH, TEST_GROUP_SIZE
    dict_red = os.path.join(DATA_PATH, "train_red")
    dict_green = os.path.join(DATA_PATH, "train_green")
    dict_blue = os.path.join(DATA_PATH, "train_blue")
    dict_mask = f"{DATA_PATH}train_gt/"
    
    if not all(os.path.exists(p) for p in [dict_red, dict_green, dict_blue, dict_mask]):
        sys.exit("ERROR: One or more dataset directories do not exist.")
    
    files_gt = os.listdir(dict_mask)
    TEST_GROUP_SIZE = min(TEST_GROUP_SIZE, len(files_gt))
    random_patches = random.sample(files_gt, TEST_GROUP_SIZE)
    PATCHES, MASKS = [], []

    for count, mask_name in enumerate(random_patches):
        print("Load patch nr:", count + 1)
        path_mask = os.path.join(DATA_PATH, "train_gt", mask_name)
        path_red = path_mask.replace("train_gt/gt_", "train_red/red_")
        path_green = path_mask.replace("train_gt/gt_", "train_green/green_")
        path_blue = path_mask.replace("train_gt/gt_", "train_blue/blue_")

        if not all(os.path.exists(p) for p in [path_red, path_green, path_blue, path_mask]):
            print(f"ERROR: Missing RGB layer or mask for {mask_name}")
            continue

        red_layer = cv2.imread(path_red, cv2.IMREAD_UNCHANGED)
        green_layer = cv2.imread(path_green, cv2.IMREAD_UNCHANGED)
        blue_layer = cv2.imread(path_blue, cv2.IMREAD_UNCHANGED)
        rgb = np.dstack((red_layer, green_layer, blue_layer))
        rgb = cv2.resize(rgb, (RESOLUTION, RESOLUTION))
        rgb = rgb.astype(np.float32)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val

        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)[..., np.newaxis]

        rgb_aug, mask_aug = augmenter_data(rgb, mask)
        PATCHES.append(rgb_aug)
        MASKS.append(mask_aug)

    PATCHES = np.array(PATCHES, dtype=np.float32)
    MASKS = np.array(MASKS, dtype=np.float32)

    return PATCHES, MASKS

# ---------------- LOSS FUNCTION ----------------
def bce_dice_loss(y_true, y_pred):
    """Combined Binary Crossentropy and Dice Loss for segmentation tasks."""
    smooth = 1.0
    vector_true = tf.keras.backend.flatten(y_true)
    vector_pred = tf.keras.backend.flatten(y_pred)
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    intersection = tf.reduce_sum(vector_true * vector_pred)
    dice_loss = 1 - (2 * intersection + smooth) / (tf.reduce_sum(vector_true) + tf.reduce_sum(vector_pred) + smooth)
    return bce + dice_loss

# ---------------- U-NET MODEL ----------------
def build_U_Net():
    """Construct a standard U-Net architecture with configurable filters and input resolution."""
    input_shape = (RESOLUTION, RESOLUTION, 3)
    filters = FILTERS
    inputs = layers.Input(input_shape)

    def conv_block(inputs, filters):
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    # Encoder
    c1 = conv_block(inputs, filters)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, filters*2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, filters*4)
    p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, filters*8)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = conv_block(p4, filters*16)

    # Decoder
    def decoder_block(last, skip_conn, filters):
        x = layers.UpSampling2D()(last)
        x = layers.concatenate([x, skip_conn])
        x = conv_block(x, filters)
        return x

    d1 = decoder_block(c5, c4, filters*8)
    d2 = decoder_block(d1, c3, filters*4)
    d3 = decoder_block(d2, c2, filters*2)
    d4 = decoder_block(d3, c1, filters)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=['accuracy'])
    return model

# ---------------- TRAINING ----------------
PATCHES, MASKS = load_data()
if len(PATCHES) == 0:
    sys.exit("ERROR: No valid files found. Check folder structure and file names.")

X_train, X_test, y_train, y_test = train_test_split(PATCHES, MASKS, test_size=TEST_SIZE, random_state=SEED)
model = build_U_Net()

print("Training U-Net...")
start = time.time()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
print(f"Training duration: {time.time()-start:.2f} sec")

if SAVE:
    print("Saving model...")
    model.save("UNet_cloud_detector_NN_model.h5")

# ---------------- POST-PROCESSING ----------------
def post_process_mask(mask_pred, rgb_image):
    """Apply morphological operations to clean up predicted masks."""
    mask = (mask_pred[..., 0] > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# ---------------- TEST & VISUALIZATION ----------------
def test_model(model, patch_index=0):
    """Visualize input RGB, ground truth mask, and predicted mask."""
    rgb_patch = PATCHES[patch_index]
    gt_mask = MASKS[patch_index]
    input_tensor = np.expand_dims(rgb_patch, axis=0)
    pred_mask_prob = model.predict(input_tensor)[0]
    pred_mask_pp = post_process_mask(pred_mask_prob, rgb_patch)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_patch)
    plt.title("Input RGB")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask[..., 0], cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_pp, cmap='gray')
    plt.title("Predicted Mask (Morphology only)")
    plt.axis('off')
    plt.show()

test_model(model)
