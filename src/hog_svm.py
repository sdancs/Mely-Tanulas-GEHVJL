import time
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


# Minden képet ekkorára méretezünk át a feldolgozás előtt
IMAGE_SIZE = (128, 128)
SEED = 14
ORIENTATIONS = 9 # különböző irányok száma
PIXELS_PER_CELL = (8, 8) #8x8-as blokkokba megyünk végig
CELLS_PER_BLOCK = (2, 2) #4 szomszédos cellából lesz a 8x8 (4x2x2)
PROGRESS_INTERVAL = 500 #konzolra, hogy áll a feldolgozás

def format_count(number):
    # kiíráshoz/olvasáshoz segítség
    return f"{number:,}".replace(",", " ")

def extract_features(dataset, name):
    total = len(dataset)#feldolgozandó képek száma

    # Kiírjuk, milyen beállításokkal indul a HOG-jellemzők kinyerése.
    print(f"\n{'=' * 60}")
    print(f"HOG-jellemzők kinyerése: {name}!")
    print(f"Feldolgozandó képek: {format_count(total)}")
    print(f"Képméret: {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} pixel")
    print(f"Élek irányainak száma: {ORIENTATIONS}")
    print(f"Egy cella mérete: {PIXELS_PER_CELL}")
    print(f"Egy blokk mérete cellákban: {CELLS_PER_BLOCK}")
    print(f"{'=' * 60}")

    start_time = time.perf_counter() #időmérés indítása
    features = None
    labels = np.empty(total, dtype=np.int64) #címkék előre lefoglalása (ugyebár a címkék számként vannak tárolva)
    for i, item in enumerate(dataset):
        image = item["image"].convert("RGB") #RGB alakítása
        image = image.resize(IMAGE_SIZE) #kép->128x128
        image_array = np.asarray(image, dtype=np.uint8)
        # HOG inicializálás
        hog_features = hog(
            image_array,  # adatokat tároló tömb
            orientations=ORIENTATIONS,  # élek csoportosítása
            pixels_per_cell=PIXELS_PER_CELL,  # fenti pixel cella és egyéb adatok
            cells_per_block=CELLS_PER_BLOCK,
            block_norm="L2-Hys",  # Ez a normalizálás mérsékli a fényerő és kontraszt hatását
            channel_axis=-1  # Megmondjuk, hogy az RGB színcsatornák az utolsó tengelyen vannak
        )
        if features is None: #első kép után már állíthatóak a features/jellemzők
            features = np.empty(
                (total, len(hog_features)),  # Soronként egy kép, oszloponként egy HOG-jellemző
                dtype=np.float32
            )
            print(f"Egy képhez tartozó HOG-jellemzők száma: {format_count(len(hog_features))}") #kiírása a jellemzők hosszának
            print(f"A teljes jellemzőmátrix mérete: ({format_count(total)}, {format_count(len(hog_features))})")
        features[i] = hog_features
        #címke átadás
        labels[i] = item["label"]
        processed = i + 1
        if processed % PROGRESS_INTERVAL == 0 or processed == total: #kiírás ugye 500 soronként
            elapsed = time.perf_counter() - start_time #idő kiírás a futásról
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / speed if speed > 0 else 0 #becslések a további futási időről
            percentage = processed / total * 100
            print(
                f"[{name}] {format_count(processed)}/{format_count(total)} kép "
                f"({percentage:.1f}%) | "
                f"Eltelt: {elapsed:.1f} | "
                f"Még kb.: {remaining:.1f} | "
                f"Tempó: {speed:.1f} kép/másodperc",
                flush=True  #pufferelés kihagyása
            )

    elapsed = time.perf_counter() - start_time
    print(f"\nKész a HOG-jellemzők kinyerése: {name}!")
    print(f"Teljes idő: {elapsed:.1f}")
    print(f"A jellemzőmátrix mérete: {features.shape}")

    # Megmutatjuk, körülbelül mennyi memóriát foglal maga a jellemzőmátrix.
    print(f"A jellemzőmátrix mérete a memóriában: {features.nbytes / (1024 ** 3):.2f} GiB")
    return features, labels


def main(original_train):
    total_start = time.perf_counter() #teljes futási idő

    # Jelezzük, hogy indul a HOG + SVM alapmodell.
    print("\n" + "#" * 60)
    print("Indul a HOG + SVM alapmodell!")
    print("#" * 60)

    # Kiírjuk a seedet, hogy később is lássuk, milyen beállítással futott a kód, mindig 14
    print(f"Használt seed: {SEED}")
    print(f"Az eredeti tanítóhalmazban {format_count(len(original_train))} kép van.")
    # A képek 80%-át tanításra, 20%-át validációra tartjuk fenn
    split_data = original_train.train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label"  # Minden kategória arányát igyekszünk megőrizni mindkét részben
    )
    train_data = split_data["train"] # a szétvágott adatok elkülönítése
    validation_data = split_data["test"]

    # Kiírjuk, hány kép jutott a két részbe (ez ugye a 75750-ből megy)
    print(f"Tanításra: {format_count(len(train_data))} kép")
    print(f"Ellenőrzésre: {format_count(len(validation_data))} kép")
    print("Az adatok felosztása kész.")
    print("\n2. lépés: kinyerjük a képek jellemzőit.")

    # Elkészítjük a tanítóképek jellemzőmátrixát és címkéit
    X_train, y_train = extract_features(
        train_data,  # Ezekből fog tanulni a modell
        "Tanítóhalmaz"  # Ezt a nevet írjuk ki a konzolra
    )

    # Elkészítjük a validációs képek jellemzőmátrixát és címkéit
    X_validation, y_validation = extract_features(
        validation_data,  # Ezeket a képeket nem használjuk fel a tanításhoz
        "Validációs halmaz"  # Ezt a nevet írjuk ki a konzolra
    )

    # Ellenőrizzük, hogy az előállított adatok mérete rendben van-e.
    print("\n3. lépés:ellenőrzés a tanítás előtt")
    print(f"Tanító jellemzők mérete: {X_train.shape}")
    print(f"Tanító címkék mérete: {y_train.shape}")
    print(f"Validációs jellemzők mérete: {X_validation.shape}")
    print(f"Validációs címkék mérete: {y_validation.shape}")
    print(f"Tanító kategóriák száma: {len(np.unique(y_train))}") #megvan mind a 101?
    print("\n4. lépés: jöhet az SVM tanítása, sok időbe telhet")
    model_start = time.perf_counter() #külön tanítási idő mérés

    model = LinearSVC( #lineáris osztályozás
        C=1.0,  # A C szabályozza a hibás besorolások és az egyszerűbb modell közötti kompromisszumot
        max_iter=3000,  # Legfeljebb ennyi optimalizálási iterációt engedünk
        random_state=SEED,  # A véletlenmagot itt is egységesen 14-re állítjuk
        verbose=0  # A scikit-learn több tanítási állapotot is kiír a konzolra
    )

    model.fit(X_train, y_train)
    model_elapsed = time.perf_counter() - model_start

    # Kiírjuk a tanítás eredményének fontosabb adatait.
    print(f"\nAz SVM tanítása kész! Idő: {model_elapsed:.1f}")
    print(f"Iterációk száma: {model.n_iter_}")

    print("\n5. lépés: validációs teszt")
    evaluation_start = time.perf_counter() #validációs időmérés is megy

    predictions = model.predict(X_validation) #jóslás/becslés a képekre

    # Az Accuracy megmutatja, a képek mekkora részét sorolta be helyesen a modell:
    accuracy = accuracy_score(y_validation, predictions)

    # A Macro-F1 kategóriánként értékel, majd az eredményeket egyenlő súllyal átlagolja.
    macro_f1 = f1_score(
        y_validation,  # A képek valódi kategóriái
        predictions,  # A modell által megjósolt kategóriák
        average="macro"  # Minden ételkategória ugyanakkora súlyt kap
    )

    evaluation_elapsed = time.perf_counter() - evaluation_start
    total_elapsed = time.perf_counter() - total_start #teljes futás számolása

    print("\n" + "=" * 60)
    print("HOG + SVM eredménye:")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Értékelés ideje: {evaluation_elapsed:.1f}")
    print(f"Teljes futási idő: {total_elapsed:.1f}")
    print("=" * 60)

    # Visszaadjuk a betanított modellt, hogy később a külön teszthalmazon is kipróbálhassuk
    return model

if __name__ == "__main__":
    pass