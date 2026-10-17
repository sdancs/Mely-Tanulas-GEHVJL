Food 101 képek osztályozása mélytanulással

A projekt célja olyan képosztályozó létrehozása, amely adott ételről készült kép alapján megpróbálja meghatározni, hogy milyen kategóriába tartozik. A megoldáshoz a nyilvánosan elérhető Food 101 adathalmazt használom fel, majd különböző módszerekkel elemzem.

Fő feladatrészek:
-bemenet: ételt ábrázoló digitális kép
-kimenet: felismert ételkategória a hozzá tartozó % valószínűséggel
-típus: felügyelt, többosztályos kép felismerés és osztályozás

adatforrás: https://huggingface.co/datasets/ethz/food101

Az adathalmaz 101000 darab képet tartalmaz 101 ételkategóriában. Kategóriánként 750 tanítókép és 250 tesztkép érhető el.

Adatelőkészítés:
-Képek betöltése
-Egységes méretre alakítás
-Adatok normalizálása
-Augmentáció (képek torzítása)

Modellválasztás
A feladat megoldására három, eltérő megközelítésen alapuló modellt választunk: egy hagyományos gépi tanulási módszert alkalmazó HOG + SVM osztályozót, egy saját fejlesztésű konvolúciós neurális hálózatot (CNN), valamint egy előtanított ResNet18 modellt.
A három megoldás összehasonlításával nemcsak azt vizsgálhatjuk meg, hogy melyik modell teljesít jobban az ételképek felismerésében, hanem azt is, hogy milyen előnyt jelent a mélytanulás alkalmazása a hagyományos képfeldolgozással szemben, illetve mennyire hasznosak a más adathalmazon előzetesen megtanult képi jellemzők (főleg a ResNet esetében).
