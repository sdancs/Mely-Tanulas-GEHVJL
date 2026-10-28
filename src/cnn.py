import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True #ha a kép valamiért nem olvasható, ne kezdődjön az egész epoch 0-ról


# A képeket egységesen 396x396 pixel méretűre alakítjuk, kissé kisebb
IMAGE_SIZE = 396

# Egy tanítási lépésben ennyi képet dolgoz fel a modell
BATCH_SIZE = 64

# Hányszor haladjon végig a modell a teljes tanító adathalmazon
EPOCHS = 130
# A tanulási ráta határozza meg a súlyok módosításának mértékét
LEARNING_RATE = 0.001
# A véletlenszerű folyamatok ismételhetőbbé tétele
SEED = 14
# A tanító adathalmaz 20%-át belső validációra használjuk:
VALIDATION_RATIO = 0.20
# Az eredmények a projekt gyökérkönyvtárában ide kerülnek.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eredmenyek" / "cnn"

def set_seed(seed=SEED):
    """Rögzíti a véletlenmagokat az ismételhetőbb futáshoz."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FoodDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self): #képek száma
        return len(self.data)

    def __getitem__(self, index): #képek címkéi

        # Kivesszük az adott képet és a hozzá tartozó kategóriát:
        row = self.data[index]

        # Minden képet RGB formátumúvá alakítunk
        image = row["image"].convert("RGB")

        # A címke egy egész szám 0 és 100 között, hisz számokkal vannak tárolva
        label = int(row["label"])

        # Elvégezzük a képfeldolgozási lépéseket:
        if self.transform is not None:
            image = self.transform(image)

        return image, label


        # A konvolúciós rétegek kinyerik a képi jellemzőket
class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # 1. blokk
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 2. blokk
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 3. blokk
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # A kimenet egységes méretre alakítása
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.features(x) #korábbi függvény behívás
        return self.classifier(x) #101 katból melyikbe tartozik

def run_epoch(
    model,
    loader,
    loss_function,
    device,
    optimizer=None,
    description=""
):
#epoch futtatása

    training = optimizer is not None

    # Tanításkor a dropout aktív
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.set_grad_enabled(training):

        for images, labels in tqdm(
            loader,
            desc=description,
            leave=False
        ):

            # Az adatokat a kiválasztott eszközre helyezzük (CPU vagy GPU)
            images = images.to(device)
            labels = labels.to(device)

            if training:
                # Az előző lépés gradienseit töröljük., kevés a memória :(
                optimizer.zero_grad()

            # A modell előrejelzései
            outputs = model(images)

            # A becsült eredményt összehasonlítjuk a valódi címkékkel
            loss = loss_function(outputs, labels)

            if training:
                # Visszaterjesztjük a hibát a hálózaton
                loss.backward()

                # Frissítjük a hálózat súlyait:
                optimizer.step()

            # A batch hibáját a képek számával súlyozzuk:
            total_loss += loss.item() * labels.size(0)

            # A legnagyobb pontszámú kategória lesz a becslés
            predictions = outputs.argmax(dim=1)

            # Az eredményeket CPU-n végezzük
            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )
            all_labels.extend(
                labels.detach().cpu().numpy()
            )

    # Az összes képre számított átlagos hiba
    average_loss = total_loss / len(loader.dataset)

    # A helyesen besorolt képek aránya:
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    # A 101 kategória F1-értékének egyszerű átlaga
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1

def main(dataset):
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # A CUDA-kompatibilis PyTorch felületet az AMD GPU-s build is használhatja
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

        # A PyTorch CPU-szálainak beállítása
    torch.set_num_threads(16)
    torch.set_num_interop_threads(8)

    print(f"Használt eszköz: {device}")
    # Az eredeti adathalmaz két részre van bontva:
    original_train = dataset["train"]
    final_test = dataset["validation"]

    # Az eredeti tanító rész 20%-át elkülönítjük validációra,m a maradék majd a teszt lesz
    # ezért a különálló eredeti validation részhez még nem nyúlunk
    split_data = original_train.train_test_split(
        test_size=VALIDATION_RATIO,
        seed=SEED,
        stratify_by_column="label"
    )

    train_data = split_data["train"]
    validation_data = split_data["test"]

    print(f"Tanítóképek: {len(train_data)}")
    print(f"Belső validációs képek: {len(validation_data)}")
    print(f"Végső tesztképek: {len(final_test)}")

    # Data augmentation: csak a tanítóképeket módosítjuk véletlenszerűen:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # A validációs és tesztképeket nem módosítjuk véletlenszerűen
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # Mindhárom adathalmaz külön Dataset objektumot kap
    train_dataset = FoodDataset(train_data, train_transform)
    validation_dataset = FoodDataset(validation_data, eval_transform)
    test_dataset = FoodDataset(final_test, eval_transform)

    # A DataLoader batch-ekben adja át a képeket a modellnek
    # A shuffle csak a tanító adatoknál aktív
    # Windows alatt a num_workers=0 egyszerűbb beállítás
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # A kategóriák nevét az adathalmazból olvassuk ki:
    class_names = dataset["train"].features["label"].names
    number_of_classes = len(class_names)

    # Létrehozzuk a modellt és áthelyezzük a CPU-ra vagy GPU-ra
    model = SimpleCNN(
        num_classes=number_of_classes
    ).to(device)

    # A CrossEntropyLoss többosztályos osztályozáshoz használhat
    loss_function = nn.CrossEntropyLoss()

    # Az Adam optimalizáló módosítja a modell súlyait
    # Súlycsökkenést is alkalmazó optimalizáló
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    # Ha a validációs pontosság nem javul, csökkentjük a tanulási rátát
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=0.000001
    )

    # A fontosabb eredményfájlok helye:
    model_path = OUTPUT_DIR / "cnn_legjobb_modell.pth"
    checkpoint_path = OUTPUT_DIR / "cnn_folytatas.pth"
    history_path = OUTPUT_DIR / "tanitasi_tortenet.csv"
    config_path = OUTPUT_DIR / "beallitasok.json"

    # futási beállítások
    config = {
        "modell": "sajat_cnn",
        "kepmeret": IMAGE_SIZE,
        "batch_meret": BATCH_SIZE,
        "epochok": EPOCHS,
        "tanulasi_rata": LEARNING_RATE,
        "seed": SEED,
        "belso_validacio_aranya": VALIDATION_RATIO,
        "tanito_kepek": len(train_data),
        "belso_validacios_kepek": len(validation_data),
        "vegso_tesztkepek": len(final_test),
        "osztalyok_szama": number_of_classes,
        "eszkoz": str(device)
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4
        )

    history = []
    best_validation_accuracy = -1.0
    start_epoch = 1

    # Ha van korábbi teljes mentés, abból folytatjuk
    if checkpoint_path.exists():
        print(f"Korábbi tanulási állapot betöltése: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        # Ellenőrizzük, hogy a mentés ugyanahhoz a modellbeállításhoz tartozik-e
        saved_settings = checkpoint.get("settings", {})
        current_settings = {
            "kepmeret": IMAGE_SIZE,
            "batch_meret": BATCH_SIZE,
            "tanulasi_rata": LEARNING_RATE,
            "seed": SEED,
            "belso_validacio_aranya": VALIDATION_RATIO,
            "osztalyok_szama": number_of_classes
        }

        if saved_settings and saved_settings != current_settings:
            raise ValueError(
                "A mentés beállításai eltérnek a jelenlegi beállításoktól. "
                "Állítsd vissza az eredeti beállításokat, vagy nevezd át/töröld "
                "a cnn_folytatas.pth fájlt, ha új tanítást szeretnél.\n"
                f"Mentett beállítások: {saved_settings}\n"
                f"Jelenlegi beállítások: {current_settings}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        history = checkpoint["history"]
        best_validation_accuracy = checkpoint["best_validation_accuracy"]
        start_epoch = checkpoint["epoch"] + 1

        # A véletlenszerű folyamatok állapotának visszaállítása
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])

        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

        print(
            f"Folytatás a(z) {start_epoch}. epochtól "
            f"({checkpoint['epoch']} epoch már befejeződött)"
        )
    else:
        print("Nem található korábbi teljes mentés, új tanítás indul.")

    print("\nCNN tanítása indul...")
    for epoch in range(start_epoch, EPOCHS + 1):

        # A modell megtanulja a tanítóképek jellemzőit
        train_loss, train_accuracy, train_f1 = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            description=f"{epoch}/{EPOCHS} tanítás"
        )

        # Megnézzük, hogyan teljesít a korábban nem tanított validációs képeken
        val_loss, val_accuracy, val_f1 = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
            description=f"{epoch}/{EPOCHS} validáció"
        )

        # A tanulási ráta igazítása a validációs pontosság alapján
        scheduler.step(val_accuracy)

        print(
            f"Aktuális tanulási ráta: "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        # Az epoch eredményeit eltároljuk:
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1
        }

        history.append(row)

        # Minden epoch után mentjük a tanítás történetét:
        # Így egy megszakítás esetén az eddigi mérések megmaradnak.
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )

            writer.writeheader()
            writer.writerows(history)

        print(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"Tanítási loss: {train_loss:.4f}\n"
            f"Tanítási pontosság: {train_accuracy:.4f}\n"
            f"Belső validációs loss: {val_loss:.4f}\n"
            f"Belső validációs pontosság: {val_accuracy:.4f}\n"
            f"Belső validációs macro F1: {val_f1:.4f}"
        )

        # A legjobb modellt a belső validációs pontosság alapján választódik
        if val_accuracy > best_validation_accuracy:

            best_validation_accuracy = val_accuracy

            # Nemcsak a súlyokat, hanem a szükséges metaadatokat
            # is elmentjük a későbbi betöltéshez:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "number_of_classes": number_of_classes,
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "best_validation_accuracy":
                        best_validation_accuracy,
                    "epoch": epoch
                },
                model_path
            )

            print("A legjobb modell elmentve.")

        # A teljes tanulási állapot mentése minden befejezett epoch után
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_validation_accuracy": best_validation_accuracy,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "settings": {
                "kepmeret": IMAGE_SIZE,
                "batch_meret": BATCH_SIZE,
                "tanulasi_rata": LEARNING_RATE,
                "seed": SEED,
                "belso_validacio_aranya": VALIDATION_RATIO,
                "osztalyok_szama": number_of_classes
            }
        }

        # Ideiglenes fájlba mentünk, hogy félbeszakadáskor a régi mentés megmaradjon
        temporary_checkpoint_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint_data, temporary_checkpoint_path)
        temporary_checkpoint_path.replace(checkpoint_path)
        print(f"Teljes tanulási állapot mentve: {checkpoint_path}")

    # Visszatöltjük a belső validáción legjobban teljesítő modellt
    # ezért is végső teszthalmaz eredménye nem befolyásolja a modellválasztást
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "\nVégső tesztelés az eredeti validation adatokon"
    )

    test_loss, test_accuracy, test_macro_f1 = run_epoch(
        model,
        test_loader,
        loss_function,
        device,
        optimizer=None,
        description="Végső teszt"
    )

    # tesztkép előrejelzések
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(
            test_loader,
            desc="Tesztpredikciók"
        ):

            outputs = model(images.to(device))

            # A legnagyobb pontszámú kategória a becslés
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    # Részletes kiértékelés kategóriánként:
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(number_of_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Összefoglaljuk a végső teszt eredményeit
    metrics = {
        "modell": "sajat_cnn",
        "legjobb_epoch": int(checkpoint["epoch"]),
        "belso_validacios_legjobb_pontossag": float(
            checkpoint["best_validation_accuracy"]
        ),
        "teszt_loss": float(test_loss),
        "teszt_pontossag": float(test_accuracy),
        "teszt_macro_f1": float(test_macro_f1),
        "tesztkepek_szama": len(test_dataset),
        "osztalyok_szama": number_of_classes
    }
    metrics_path = OUTPUT_DIR / "vegso_teszteredmeny.json"
    report_path = OUTPUT_DIR / "osztalyonkenti_eredmenyek.json"
    predictions_path = OUTPUT_DIR / "teszt_predikciok.csv"

    # Végső mérőszámok mentés
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Osztályonkénti értékek mentése különk ülön
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Képenként elmentjük a valódi és becsült kategóriát
    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "valos_cimke",
            "becsult_cimke",
            "helyes"
        ])

        for true_label, predicted_label in zip(
            all_labels,
            all_predictions
        ):

            writer.writerow([
                class_names[int(true_label)],
                class_names[int(predicted_label)],
                int(true_label == predicted_label)
            ])


    print("\nVégső teszteredmények:")
    print(f"Teszt loss: {test_loss:.4f}")
    print(f"Teszt pontosság: {test_accuracy:.4f}")
    print(f"Teszt macro F1: {test_macro_f1:.4f}")

    print(f"\nEredmények mappája: {OUTPUT_DIR.resolve()}")
    print(f"Legjobb modell: {model_path}")
    print(f"Tanítási történet: {history_path}")
    print(f"Végső eredmények: {metrics_path}")
    print(f"Osztályonkénti eredmények: {report_path}")
    print(f"Tesztpredikciók: {predictions_path}")
