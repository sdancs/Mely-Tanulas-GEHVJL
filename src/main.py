
from datasets import load_dataset
from hog_svm import main as hog_svm_main
from cnn import main as cnn_main


def main():
    print("Food-101 adathalmaz betöltése...")
    dataset = load_dataset("ethz/food101")

    train_data = dataset["train"]
    validation_data = dataset["validation"]

    print(f"Tanítóképek száma: {len(train_data)}")
    print(f"Validációs képek száma: {len(validation_data)}")
    print(f"Összes kép száma: {len(train_data) + len(validation_data)}")

    first_image = train_data[0]["image"]
    first_label = train_data[0]["label"]
    label_names = train_data.features["label"].names

    print(f"Első kép címkéje: {first_label}")
    print(f"Első kép kategóriája: {label_names[first_label]}")
    print(f"Első kép mérete: {first_image.size}")

    # HOG + SVM futtatása
    # hog_svm_main(train_data)

    # CNN futtatása
    cnn_main(dataset)


if __name__ == "__main__":
    main()
