from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import shuffle
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
import argparse
import os


def get_data(directory_path, image_size=(128, 128)):
    """
    Load spectrogram images from a specified directory and resize them.

    :param directory_path: Path to the directory containing spectrogram images.
    :param image_size: A tuple indicating the size to which images should be resized.
    :return: A NumPy array of images.
    """
    images = []
    # Iterate over files in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        # Check if file is an image
        if os.path.isfile(file_path) and file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            # Open the image
            with Image.open(file_path) as img:
                # Resize image and convert to grayscale
                img = img.resize(image_size).convert("L")
                # Convert image to numpy array and normalize
                img_array = np.array(img) / 255.0
                images.append(img_array)

    # Convert list of images to a NumPy array
    return np.array(images)


def define_model():
    """
    Define and return the CNN model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(128, 128, 1)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def lr_schedule(epoch):
    """
    Learning rate schedule function.
    """
    lr = 1e-4
    if epoch > 5:
        lr *= 0.1
    if epoch > 10:
        lr *= 0.1
    return lr


def compile_model(model):
    """
    Compile the CNN model.
    """
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, patience):
    """
    Train the model and return the training history.
    """
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        )
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print metrics.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)


def plot_metrics(history):
    """
    Plot training and validation loss and accuracy.

    :param history: Training history object from Keras model.
    :return: None
    """
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss by Epoch", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Crossentropy", fontsize=12)
    plt.xticks(range(0, 20, 1), range(0, 20, 1))
    plt.legend(fontsize=12)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy by Epoch", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(range(0, 20, 1), range(0, 20, 1))
    plt.legend(fontsize=12)

    plt.show()


def train_evaluate_model(
    X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, patience=20
):
    """
    Train and evaluate a convolutional neural network model.
    """
    model = define_model()
    model = compile_model(model)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs, patience)
    model = tf.keras.models.load_model("best_model.h5")
    evaluate_model(model, X_test, y_test)
    plot_metrics(history)


def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN model on spectrogram images."
    )
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing spectrogram images.",
    )
    args = parser.parse_args()

    # Load and preprocess data
    images = get_data(args.directory_path)
    images = np.array(images)
    images = images.reshape(images.shape[0], 128, 128, 1)
    labels = np.zeros((images.shape[0],))

    # Set the labels dynamically based on a percentage of the total images
    percentage_for_label_1 = 0.25  # For example, 25%
    num_labels_1 = int(len(images) * percentage_for_label_1)
    labels[:num_labels_1] = 1

    # Shuffle the data
    images, labels = shuffle(images, labels, random_state=42)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train and evaluate the model
    train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save the model
    model = tf.keras.models.load_model("best_model.h5")
    model.save("model.h5")


if __name__ == "__main__":
    main()
