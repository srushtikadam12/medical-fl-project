import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator

# Function for data normalization

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

# Function for data augmentation

def augment_data(images):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented_images = [datagen.random_transform(image) for image in images]
    return augmented_images

# Example usage of the functions
#if __name__ == '__main__':
#    data = np.random.rand(100, 64, 64, 3)  # Placeholder for medical images
#    normalized = normalize_data(data.reshape(-1, data.shape[-1])).reshape(data.shape)
#    augmented = augment_data(normalized)