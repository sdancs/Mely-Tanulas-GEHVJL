from datasets import load_dataset
<<<<<<< Updated upstream
=======
from hog_svm import main as hog_svm_main
from cnn import main as cnn_main
>>>>>>> Stashed changes


def main():

    print("Food-101 adathalmaz betöltése...")
    #Huggingface hivatkozása:
    dataset = load_dataset("ethz/food101")
    # A képek két külön részre vannak bontva:
    # A train a tanításhoz, a validation pedig az ellenőrzéshez kell majd
    train_data = dataset["train"]
    validation_data = dataset["validation"]

    print(f"Tanítóképek száma: {len(train_data)}")
    print(f"Validációs képek száma: {len(validation_data)}")
    print(f"Összes kép száma: {len(train_data) + len(validation_data)}")

    # Első kép: ellenőrizzük, hogy a kép és a címke is rendben van-e
    first_image = train_data[0]["image"]
    first_label = train_data[0]["label"]

    # A címke egy szám, ebből kérjük le az emberi szemmel is érthető nevet:
    label_names = train_data.features["label"].names
    category_name = label_names[first_label]

    print(f"Első kép címkéje: {first_label}")
    print(f"Első kép kategóriája: {category_name}")
    print(f"Első kép mérete: {first_image.size}")

<<<<<<< Updated upstream
    # Mutassuk is meg az első képet, hogy lássuk, tényleg jó adatot kaptunk:
    first_image.show()
=======
    #first_image.show()

    #HOG-SVM futtatása:
    #hog_svm_main(train_data)
    #CNN futtatása
    cnn_main(dataset)
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
<<<<<<< Updated upstream
=======


#.venv\Scripts\activate.bat
>>>>>>> Stashed changes
