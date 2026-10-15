from collections import Counter
from datasets import load_dataset


def inspect_split(data, split_name, label_names):
    # Nézzük meg, hogy minden rendben van-e az adatokkal:
    valid_images = 0
    missing_images = 0
    invalid_images = 0
    invalid_labels = 0

    image_sizes = []
    label_counts = Counter() # counter -> elemek megszámlálása

    # Végigmegyünk az összes képen, és ellenőrizzük őket:
    for index in range(len(data)):
        try:
            sample = data[index]
            image = sample["image"]
            label = sample["label"]

            # Megnézzük, hogy a kép tényleg megvan-e:
            if image is None:
                missing_images += 1
                continue

            # A kép mérete sem lehet nulla:
            if image.width <= 0 or image.height <= 0:
                invalid_images += 1
                continue

            # A címkének a 101 kategória valamelyikéhez kell tartoznia
            if not isinstance(label, int) or not (0 <= label < len(label_names)):
                invalid_labels += 1
                continue

            valid_images += 1
            image_sizes.append(image.size)
            label_counts[label] += 1

        except Exception as error:
            # Ha valami nem olvasható be rendesen
            invalid_images += 1
            print(f"Hibás rekord a(z) {index}. indexnél: {error}")

    print(f"\n--- {split_name} adatok ellenőrzése ---")
    print(f"Összes rekord: {len(data)}")
    print(f"Érvényes képek: {valid_images}")
    print(f"Hiányzó képek: {missing_images}")
    print(f"Hibás képek: {invalid_images}")
    print(f"Hibás címkék: {invalid_labels}")

    # Megnézzük, mekkorák a képek:
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]

        print(f"Legkisebb képméret: {min(widths)} x {min(heights)}")
        print(f"Legnagyobb képméret: {max(widths)} x {max(heights)}")

    # Megnézzük, hogy mind a 101 kategóriában van-e kép:
    missing_categories = [
        label_names[label]
        for label in range(len(label_names))
        if label_counts[label] == 0
    ]

    print(f"Előforduló kategóriák: {len(label_counts)}")

    if missing_categories:
        print("Képek nélküli kategóriák:", missing_categories)
    else:
        print("Minden kategóriához tartozik kép.")

    return {
        "valid_images": valid_images,
        "missing_images": missing_images,
        "invalid_images": invalid_images,
        "invalid_labels": invalid_labels,
        "label_counts": label_counts,
    }


def main():

    print("Food-101 adathalmaz betöltése")

    # Huggingface hivatkozása
    dataset = load_dataset("ethz/food101")

    # A képek két külön részre vannak bontva:
    # A train a tanításhoz, a validation pedig az ellenőrzéshez kell majd
    train_data = dataset["train"]
    validation_data = dataset["validation"]

    # A címke egy szám, ebből kérjük le az emberi szemmel is érthető nevet:
    label_names = train_data.features["label"].names
    print(f"Ételkategóriák száma: {len(label_names)}")

    # Ellenőrizzük, hogy a két adatrészben ugyanazok-e a címkék:
    validation_label_names = validation_data.features["label"].names

    if label_names != validation_label_names:
        raise ValueError("A train és validation címkekészlete eltér!")

    # Nézzük meg, hogy rendben vannak-e a tanítóképek:
    train_report = inspect_split(
        train_data, "Tanítóhalmaz", label_names
    )

    # Ugyanezt megnézzük a validációs képeknél is:
    validation_report = inspect_split(
        validation_data, "Validációs halmaz", label_names
    )

    # Összesítjük, hogy találtunk-e valamilyen problémát:
    total_missing = (
        train_report["missing_images"]
        + validation_report["missing_images"]
    )

    total_invalid_images = (
        train_report["invalid_images"]
        + validation_report["invalid_images"]
    )

    total_invalid_labels = (
        train_report["invalid_labels"]
        + validation_report["invalid_labels"]
    )

    print("\n--- Adatminőségi összesítés ---")
    print(f"Hiányzó képek: {total_missing}")
    print(f"Hibás képek: {total_invalid_images}")
    print(f"Hibás címkék: {total_invalid_labels}")

    if (
        total_missing == 0
        and total_invalid_images == 0
        and total_invalid_labels == 0
    ):
        print("Az ellenőrzött adatok rendben vannak.")
    else:
        print("Problem van!")


if __name__ == "__main__":
    main()
