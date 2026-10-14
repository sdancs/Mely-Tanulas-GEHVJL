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