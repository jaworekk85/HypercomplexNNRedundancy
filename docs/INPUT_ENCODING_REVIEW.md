# Kodowanie wejścia w pracach Niemczynowicz i Kyci

Sprawdzono: 2026-11-02. Klucze cytowań pochodzą z `references.bib`.
To przegląd źródeł i decyzji o kodowaniu; nie wykonano nowego eksperymentu.

## Ustalenia ze źródeł

### Obrazy: `NiemczynowiczKycia2025`

W [preprincie KHNN, sekcja 3](https://arxiv.org/html/2407.00452v1#S3),
przykład klasyfikacji malarii wykorzystuje obrazy RGB. Autorzy opisują dodanie
zerowego kanału alfa, przypisanego części rzeczywistej, czyli jeden piksel jako
cztery składowe. Opisaną konwencję ARGB można zapisać jako
`q(h,w) = 0 + R(h,w)i + G(h,w)j + B(h,w)k`.

W [notebooku ExampleConvolutional.ipynb](https://github.com/rkycia/HypercomplexKeras/blob/566f887a7039637860089863b5abfa13b1948267/examples/ExampleConvolutional.ipynb)
jest jednak konwersja RGB do RGBA, a następnie odwrócenie całej osi kanałów.
[Implementacja TensorFlow I/O v0.31.0](https://github.com/tensorflow/io/blob/v0.31.0/tensorflow_io/python/experimental/color_ops.py)
dodaje alfa równe zero. Złożenie tych operacji daje **ABGR: `(0,B,G,R)`**, czyli
`q = 0 + Bi + Gj + Rk`. Jest to wniosek z kodu, a nie pomiar treningu.
Komentarz notebooka również wskazuje ABGR. Nie należy utożsamiać tego
porządku z ARGB opisanym w tekście.

### Szeregi czasowe: `KyciaNiemczynowicz2025`

W opublikowanej pracy [Scientific Reports, Methods i Results](https://www.nature.com/articles/s41598-025-08957-5)
czterema składowymi są wartości czterech powiązanych serii z tej samej chwili:
**Copper, FCX, CLP, SCCO**. W wariancie kwaternionowym to przypisanie odpowiada
`q_t = Copper_t + FCX_t i + CLP_t j + SCCO_t k`.
Wejście obejmuje okno historii; celem jest przyszły przebieg Copper.

Autorzy badali też kolejność **FCX, CLP, SCCO, Copper**, co przenosi Copper
z części rzeczywistej do składowej `k`. Praca dotyczy również innych algebr 4D;
nie należy nazywać wszystkich jej modeli kwaternionowymi. W równaniu (3)
powtarza się indeks pierwszej składowej; powyższe przypisanie odtwarzamy
z opisu czterech odrębnych serii i ich kolejności.

### Ogólny format: `NiemczynowiczKycia2026`

[Preprint tensorowy, sekcje 3.1–3.2](https://arxiv.org/html/2407.00449v1#S3)
opisuje pakowanie kanałów piksela w element algebry. Dla Dense ostatnia oś
ma rozmiar będący wielokrotnością wymiaru algebry.
[Kod HyperDense](https://github.com/rkycia/HypercomplexKeras/blob/566f887a7039637860089863b5abfa13b1948267/src/HypercomplexKeras/Hyperdense.py)
oczekuje bloków składowych: `[r_1,...,r_U,i_1,...,i_U,j_1,...,j_U,k_1,...,k_U]`.
Nasza warstwa lokalna ma jawny układ `(N,U,4)`; konwersja między tymi układami
wymaga permutacji osi, a nie samego spłaszczenia.

## Interpretacja i następny wybór

Wspólny motyw tych przykładów to powiązane kanały jednej obserwacji: jednego
piksela albo jednego czasu. W sprawdzonych źródłach nie znalazłem uzasadnienia
wcześniej zaproponowanego kodowania MNIST blokami `2×2`; była to nasza
propozycja, a nie odtworzenie rozwiązania autorów.

Bliższym ich przykładowi obrazowemu punktem startowym byłyby małe dane RGB,
z jawną konwencją `(0,R,G,B)` lub odtworzeniem `(0,B,G,R)` z notebooka.
Dla lokalnego Dense jeden piksel byłby jedną jednostką wejściową, dając
`(N,H*W,4)`. Zera nie są dodatkową informacją; kontrola rzeczywista powinna
otrzymać te same dane RGB. To uzasadnienie wyboru reprezentacji, nie dowód
jej przewagi. MNIST `2×2` pozostaje możliwym wariantem do osobnego sprawdzenia.

## Zakres sprawdzenia

Sprawdzono sekcje dotyczące danych i reprezentacji wejścia w Scientific Reports
oraz preprintach KHNN i pracy tensorowej, a także notebooki i kod
HypercomplexKeras w commicie
`566f887a7039637860089863b5abfa13b1948267`. Strona wydawcy SoftwareX była
niedostępna (HTTP 403); nie twierdzimy, że porównano ostateczny tekst z preprintem.
Nie uruchamiano notebooka obrazowego i nie zmieniano konfiguracji Phase 0.
