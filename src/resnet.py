import json
import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader  #A képek batch-ekben történő betöltése
from torchvision import transforms  #Képtranszformációk a tanításhoz
from torchvision.models import resnet101, ResNet101_Weights  #ImageNet előtanított hálózata
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  #Kiértékelési metrikák
from tqdm import tqdm  #Tanítási folyamatjelző
from pathlib import Path  #Fájl kezelés


# A tanítás fő beállításai
SEED = 14
BATCH_SIZE = 32 
EPOCHS = 10 
IMAGE_SIZE = 224
NUM_CLASSES = 101
NUM_WORKERS = 0  # Windows alatt stabil adatbetöltés
OUTPUT_DIR = Path("eredmenyek") / "resnet101"  #a resnet eredményeinek mappája
MODEL_PATH = OUTPUT_DIR / "resnet101_legjobb_modell.pth"  # A legjobb modell mentési útvonala
HISTORY_PATH = OUTPUT_DIR / "tanitasi_tortenet.json"  #tanítási előzmények mentési útvonala
CONFIG_PATH = OUTPUT_DIR / "beallitasok.json"  #beállítások mentési útvonala


def main(dataset):  # A main.py által betöltött Food-101 dataset átvétele
    torch.backends.cudnn.enabled = False #AMD GPU életérzései
    torch.manual_seed(SEED)  # PyTorch véletlenszám-generátorának rögzítése
    if torch.cuda.is_available():  #AMD GPU életérzése
        torch.cuda.manual_seed_all(SEED)  #GPU véletlenszám-generátorának rögzítése
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA vagy CPU kiválasztása
    print(f"Használt eszköz: {device}")  # A használt eszköz kiírása
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  #mappa létrehozás

    original_train = dataset["train"]  # Az eredeti Food-101 tanító split átvétele
    split = original_train.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")  # 80/20 belső felosztás
    train_data = split["train"]  # 60 600 tanítókép kiválasztása
    internal_validation_data = split["test"]  # 15 150 belső validációs kép kiválasztása
    final_test_data = dataset["validation"]  # Az eredeti 25 250 kép csak a végső teszteléshez

    weights = ResNet101_Weights.DEFAULT  #ImageNeten előtanított súlyok kiválasztása
    model = resnet101(weights=weights)  #előtanított resnet betöltése
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  #!!az ImageNet 1000 osztályos fej cseréje 101 osztályosra
    model = model.to(device)  #a modellt a kiválasztott eszközre helyezzük

    # Először csak a kimeneti osztályozóréteget tanítjuk
    for parameter in model.parameters():  # Végigmegyünk a modell összes paraméterén
        parameter.requires_grad = False  # Alapértelmezés szerint befagyasztjuk a paramétereket
    for parameter in model.fc.parameters():  # Kiválasztjuk az új osztályozóréteg paramétereit
        parameter.requires_grad = True  # Az osztályozóréteg tanítható marad
    ##1000-->101 osztály csere

    # Tanítási augmentációk és validációs előfeldolgozás
    train_transform = transforms.Compose([  # A tanító képek transzformációinak összeállítása
        transforms.RandomResizedCrop(IMAGE_SIZE),  #véletlenszerű kivágás és átméretezés
        transforms.RandomHorizontalFlip(),  #véletlenszerű vízszintes tükrözés
        transforms.ToTensor(),  #pillow képből tensor készítése
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),  #ImageNet normalizálás
    ])
    eval_transform = weights.transforms()  # A validációhoz és teszteléshez imagenet transzform

    def make_collate(transform):  # Batch-összeállító függvény létrehozása adott transzformációval
        def collate_fn(batch):  # A batch elemeinek feldolgozása
            images = [transform(item["image"].convert("RGB")) for item in batch]  # Képek előfeldolgozása, RGB formátumra alakítás
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  # Címkék tensorba rendezése -> adat + címke
            return torch.stack(images), labels  # Képek és címkék visszaadása
        return collate_fn  # Az elkészült batch-függvény visszaadása

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=make_collate(train_transform))  #Tanító DataLoader
    val_loader = DataLoader(internal_validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=make_collate(eval_transform))  #Belső validáció DataLoader
    test_loader = DataLoader(final_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=make_collate(eval_transform))  # Végső teszt DataLoader, valós validáció

    criterion = nn.CrossEntropyLoss()  # Többosztályos osztályozáshoz CrossEntropyLoss
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.0001)  # Csak az fc réteg optimalizálása, weight decay->ne változzon nagy mértékekben a súlyok értékei
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)  #validációs accuracy alapján LR, 2 epochba nem változik, feleződik a mérték: csökkentése

    config = {"modell": "ResNet101 transfer learning", "kepmeret": IMAGE_SIZE, "batch_meret": BATCH_SIZE, "epochok": EPOCHS, "seed": SEED, "tanito_kepek": len(train_data), "belso_validacios_kepek": len(internal_validation_data), "vegso_tesztkepek": len(final_test_data), "osztalyok_szama": NUM_CLASSES, "eszkoz": str(device)}  # Tanítási beállítások összeállítása
    CONFIG_PATH.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")  # Beállítások mentése

    history = []  # Epochonkénti eredmények tárolása
    best_val_accuracy = 0.0  # Legjobb validációs accuracy inicializálása
    training_start = time.time()  #a teljes tanítás kezdési idejének rögzítése

    for epoch in range(EPOCHS):  # Epochok végrehajtása
        epoch_start = time.time()  # Az epoch kezdési idejének rögzítése
        model.train()  # Tanítási mód bekapcsolása
        train_loss_sum = 0.0  # Tanítási loss összegének inicializálása
        train_predictions = []  # Tanítási predikciók gyűjtése
        train_targets = []  # Tanítási valódi címkék gyűjtése

        for images, labels in tqdm(train_loader, desc=f"{epoch + 1}/{EPOCHS} tanítás"):  # Tanító batch-ek feldolgozása
            images = images.to(device)  # Képek GPU-ra helyezése
            labels = labels.to(device)  # Címkék GPU-ra helyezése
            optimizer.zero_grad()  # Korábbi gradiensek törlése
            outputs = model(images)  # Modell előrejelzése
            loss = criterion(outputs, labels)  # Loss kiszámítása
            loss.backward()  # Gradiensek számítása
            optimizer.step()  # Tanítható paraméterek frissítése
            train_loss_sum += loss.item() * images.size(0)  # Batch loss összegzése képszámmal súlyozva
            train_predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())  # Predikciók mentése
            train_targets.extend(labels.detach().cpu().tolist())  # Valódi címkék mentése

        train_loss = train_loss_sum / len(train_data)  # Átlagos tanítási loss kiszámítása
        train_accuracy = accuracy_score(train_targets, train_predictions)  # Tanítási accuracy kiszámítása
        train_macro_f1 = f1_score(train_targets, train_predictions, average="macro", zero_division=0)  # Tanítási macro F1 kiszámítása

        model.eval()  # Kiértékelési mód bekapcsolása
        val_loss_sum = 0.0  # Validációs loss összegének inicializálása
        val_predictions = []  # Validációs predikciók gyűjtése
        val_targets = []  # Validációs címkék gyűjtése


        #20% validáció
        with torch.inference_mode():  # Validáció gradiensszámítás nélkül
            for images, labels in tqdm(val_loader, desc=f"{epoch + 1}/{EPOCHS} validáció"):  # Validációs batch-ek feldolgozása
                images = images.to(device)  # Képek GPU-ra helyezése
                labels = labels.to(device)  # Címkék GPU-ra helyezése
                outputs = model(images)  # Modell előrejelzése
                loss = criterion(outputs, labels)  # Validációs loss számítása
                val_loss_sum += loss.item() * images.size(0)  # Validációs loss összegzése
                val_predictions.extend(outputs.argmax(dim=1).cpu().tolist())  # Validációs predikciók mentése
                val_targets.extend(labels.cpu().tolist())  # Validációs címkék mentése

        val_loss = val_loss_sum / len(internal_validation_data)  # Átlagos validációs loss
        val_accuracy = accuracy_score(val_targets, val_predictions)  # Validációs accuracy
        val_macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)  # Validációs macro F1
        val_precision = precision_score(val_targets, val_predictions, average="macro", zero_division=0)  # Validációs macro precision
        val_recall = recall_score(val_targets, val_predictions, average="macro", zero_division=0)  # Validációs macro recall
        scheduler.step(val_accuracy)  # Learning rate scheduler frissítése validációs accuracy alapján

        epoch_result = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "train_macro_f1": train_macro_f1, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1, "val_precision": val_precision, "val_recall": val_recall, "learning_rate": optimizer.param_groups[0]["lr"], "epoch_time_sec": time.time() - epoch_start}  # Epoch eredményeinek összeállítása
        history.append(epoch_result)  # Epoch eredményének hozzáadása az előzményekhez
        HISTORY_PATH.write_text(json.dumps(history, indent=4, ensure_ascii=False), encoding="utf-8")  # Tanítási előzmények mentése

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")  # Epoch sorszámának kiírása
        print(f"Tanítási loss: {train_loss:.4f} | accuracy: {train_accuracy:.4f} | macro F1: {train_macro_f1:.4f}")  # Tanítási metrikák
        print(f"Validációs loss: {val_loss:.4f} | accuracy: {val_accuracy:.4f} | macro F1: {val_macro_f1:.4f}")  # Validációs metrikák
        print(f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | LR: {optimizer.param_groups[0]['lr']:.7f}")  # További metrikák és LR

        if val_accuracy > best_val_accuracy:  # Ellenőrizzük, hogy javult-e a validációs accuracy
            best_val_accuracy = val_accuracy  # Legjobb accuracy frissítése
            torch.save(model.state_dict(), MODEL_PATH)  # Legjobb modell súlyainak mentése
            print("A legjobb ResNet101 modell elmentve.")

    print(f"\nTanítás befejezve. Idő: {(time.time() - training_start) / 60:.2f} perc")  # Teljes tanítási idő kiírása

    #végső validálás, a valós validálási adatokon
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Legjobb modell betöltése
    model.eval()  # Kiértékelési mód beállítása
    test_predictions = []  # Végső teszt predikcióinak tárolása
    test_targets = []  # Végső teszt címkéinek tárolása
    test_loss_sum = 0.0  # Végső teszt loss összegzése
    test_start = time.time()  # Végső teszt kezdési időpontja

    with torch.inference_mode():  # Végső teszt gradiensszámítás nélkül
        for images, labels in tqdm(test_loader, desc="ResNet101 végső teszt"):
            images = images.to(device)  # Képek GPU-ra helyezése
            labels = labels.to(device)  # Címkék GPU-ra helyezése
            outputs = model(images)  # Végső teszt előrejelzése
            loss = criterion(outputs, labels)  # Végső teszt loss számítása
            test_loss_sum += loss.item() * images.size(0)  # Végső teszt loss összegzése
            test_predictions.extend(outputs.argmax(dim=1).cpu().tolist())  # Predikciók mentése
            test_targets.extend(labels.cpu().tolist())  # Valódi címkék mentése

    test_loss = test_loss_sum / len(final_test_data)  # Végső teszt átlagos loss
    test_accuracy = accuracy_score(test_targets, test_predictions)  # Végső teszt accuracy
    test_macro_f1 = f1_score(test_targets, test_predictions, average="macro", zero_division=0)  # Végső teszt macro F1
    test_precision = precision_score(test_targets, test_predictions, average="macro", zero_division=0)  # Végső teszt macro precision
    test_recall = recall_score(test_targets, test_predictions, average="macro", zero_division=0)  # Végső teszt macro recall

    print("\n" + "=" * 60)  # Eredményblokk elválasztása
    print("RESNET101 - VÉGSŐ TESZT EREDMÉNYEI")  # Eredményblokk címe
    print("=" * 60)  # Eredményblokk elválasztása
    print(f"Tesztképek száma: {len(final_test_data)}")  # Végső teszt képszám
    print(f"Test loss: {test_loss:.4f}")  # Végső teszt loss kiírása
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")  # Végső teszt accuracy
    print(f"Macro F1: {test_macro_f1:.4f}")  # Végső teszt macro F1
    print(f"Macro precision: {test_precision:.4f}")  # Végső teszt macro precision
    print(f"Macro recall: {test_recall:.4f}")  # Végső teszt macro recall
    print(f"Tesztelési idő: {(time.time() - test_start) / 60:.2f} perc")  # Végső teszt futási ideje
    print("=" * 60)  # Eredményblokk lezárása


if __name__ == "__main__":

