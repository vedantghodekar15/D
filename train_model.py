from pathlib import Path

from tensorflow import keras


MODEL_PATH = Path(__file__).resolve().parent / "models" / "baseline_ann.keras"


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()