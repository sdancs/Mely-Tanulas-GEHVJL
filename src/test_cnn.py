import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from cnn import SimpleCNN

# Beállítások
MODEL_PATH = "eredmenyek/cnn/cnn_legjobb_modell.pth"  # A betöltendő modell elérési útja
IMAGE_SIZE = 396  # A tanítás során használt képméret
BATCH_SIZE = 64
NUM_CLASSES = 101
DEVICE = torch.device("cuda")
# Képek előfeldolgozása
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])
class Food101Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # Az eredeti adathalmaz eltárolása
    def __len__(self):
        return len(self.dataset)  # Az adathalmaz méretének visszaadása, teszt!

    def __getitem__(self, index):
        row = self.dataset[index]  #Egy kép és a hozzá tartozó címke lekérése

        image = row["image"].convert("RGB")  #Kép RGB formátumra alakítása
        label = int(row["label"])  # A kategóriaazonosító lekérése
        return transform(image), label  #Előfeldolgozott kép és címke visszaadása

def main():
    dataset = load_dataset("ethz/food101")
    validation = dataset["validation"]
    print(f"Validációs képek: {len(validation)}")
    test_dataset = Food101Dataset(validation)
    #Adatok kötegekbe rendezése
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Tesztelésnél nincs szükség keverésre
        num_workers=0  # Adatbetöltés a fő folyamatban
    )
    model = SimpleCNN(num_classes=NUM_CLASSES)
    #Korábban elmentett modell betöltése:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    #A mentés típusának megfelelő súlybetöltés
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()  #Kiértékelési mód bekapcsolása
    criterion = nn.CrossEntropyLoss()
    all_predictions = []  # Modell által előrejelzett kategóriák
    all_labels = []  # Tényleges kategóriák
    total_loss = 0.0  # Összesített veszteség
    # Kiértékelés, gradiensszámítás nélkül
    with torch.no_grad():
        # Végighaladunk a teljes validációs adathalmazon
        for images, labels in tqdm(test_loader, desc="Tesztelés"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)  #Modell előrejelzéseinek kiszámítása
            loss = criterion(outputs, labels)  #Veszteség kiszámítása
            # Köteg veszteségének hozzáadása a teljes veszteséghez
            total_loss += loss.item() * images.size(0)
            # Legnagyobb kimeneti értékű kategória kiválasztása
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
    # Kiértékelési metrikák kiszámítása
    loss = total_loss / len(test_dataset)  # Átlagos veszteség
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )  #Pontosság kiszámítása
    # Macro Precision, Recall és F1 kiszámítása
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
    print("CNN MODELL - EREDETI VALIDÁCIÓS EREDMÉNYEK")
    print("=" * 45)
    print(f"Tesztképek száma:       {len(test_dataset)}")
    print(f"Loss:                   {loss:.4f}")
    print(f"Accuracy:               {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:        {precision:.4f}")
    print(f"Macro Recall:           {recall:.4f}")
    print(f"Macro F1:               {f1:.4f}")

if __name__ == "__main__":
    main()
