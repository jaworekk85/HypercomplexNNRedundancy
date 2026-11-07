# HypercomplexNNRedundancy — IJCNN 2027

**Working title:** Activation-Based Redundancy Analysis and Compression of Hypercomplex Neural Networks

**Repozytorium Git:** kod, konfiguracje, dokumentacja, wyniki tekstowe i wykresy
są wersjonowane. Duże zbiory danych oraz pośrednie macierze i checkpointy są
wyłączone; zachowujemy natomiast wybrane modele bazowe ResNet-20 oraz indeksy
podziałów. Pełne wznowienie i weryfikacja archiwalnych eksperymentów wymagają
lokalnych danych i pozostałych checkpointów, których sam klon nie zawiera.

**Zakończone:** porównanie przycinania i kompensacji ResNet-20 na CIFAR-10.
Sieci bazowe zakończone: wybrana walidacja real90,9%, Q91,0%. Porównujemy
samodzielne przycięcie, mapy i bezpośrednie dopasowanie w obu rodzinach, przy
tych samych usuwanych kanałach w danej rodzinie; 25% i50%, po3 epoki naprawy.
[Stan](results/resnet20_compression/state.json), [raport](results/resnet20_compression/REPORT.md).
Nic się obecnie nie liczy. Skróty ponownie odczytują zakończone badanie. Zapis po warstwie i epoce; plik
`results/resnet20_compression/STOP` wstrzymuje po skończeniu warstwy lub partii.
Replikacja i PHM nie startują automatycznie.

Ustalona kolejność dalszych badań: modele ResNet -> porównanie kompresji
Q/rzeczywistej -> replikacja -> hybryda Q/rzeczywista przy wspólnym budżecie ->
mała diagnostyka PHM. CIFAR-100 pozostaje osobnym późniejszym etapem.
[Pełna kolejka z zależnościami](configs/ijcnn_research_queue.json).
Porównanie kompresji zakończone; dalsza replikacja wymaga implementacji.

Pierwszy kod hybrydy Q/rzeczywistej/PHM jest gotowy technicznie:
[opis i kontrola poprawności](results/resnet_structure_catalogue/REPORT.md).
Obejmuje dwa bloki i automatyczny wybór struktury pod budżetem. Nie został jeszcze
uruchomiony na CIFAR; czeka na replikację,
której skrypt pozostaje do przygotowania. Nie zmienia obecnego badania.

**Zakończone: modele bazowe ResNet-20 na CIFAR-10.** Kolejka dwóch modeli, zwykłego
i kwaternionowego, po 200 epok, seed2026. Parametry: **269 722 / 271 594**.
Celem było około 90% trafności; oba modele go osiągnęły. Wspólny protokół SGD,
podział danych i augmentacje; brak nowych obliczeń starych pilotów.
[Stan](results/resnet20_baselines/state.json), [raport](results/resnet20_baselines/REPORT.md).
Oba skróty wznawiają ten etap. Checkpoint po każdej epoce; plik
`results/resnet20_baselines/STOP` żąda pauzy po partii. Niedokończona epoka
zostanie powtórzona po wznowieniu. Kolejny etap przycinania ResNeta czeka na
sprawdzenie jakości modeli bazowych; CIFAR-100 pozostaje na później.

Aktualny stan badań i decyzje: [RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md).
Plan pod IJCNN 2027: [PLAN_IJCNN2027.md](docs/PLAN_IJCNN2027.md). Przekazanie pracy
między agentami (co zrobiono, konwencje, jak wznowić): [HANDOFF.md](docs/HANDOFF.md).

**Najnowszy pilot zakończony (6 listopada):** kalibracja współczynników `a` i dodatkowego
kwaternionu `mu` na usuwany kanał. Cztery warianty: dotychczasowe mapy, samo `a`,
samo `mu`, oba naraz. Jedna późna warstwa, niezmienne kanały i sześć konwolucji Q;
każdy model kończy z **19 098 parametrami**. Po tej samej trzyepokowej naprawie:
kontrola **56,7%**, samo `a` **56,3%**, samo `mu` **56,6%**, oba **56,3%**.
Wspólne dopasowanie poprawiło lokalną rekonstrukcję, ale nie jakość po naprawie.
Jeden pilot, bez potwierdzonej korzyści. Niezależna kontrola CPU przeszła;
żadne dalsze obliczenia nie są automatycznie uruchamiane.
[Stan](results/centered_maps_pilot/state.json), [raport](results/centered_maps_pilot/REPORT.md).
Wspólne dopasowanie sprawdza dokładne odzyskanie `mu` po liniowym rozwiązaniu;
nie traktujemy samego uproszczenia algebraicznego jako nowości.
Ten pilot jest zamknięty. Zapis po epoce; plik
`results/centered_maps_pilot/STOP` żąda wstrzymania po partii. Niedokończona epoka
zostanie powtórzona po wznowieniu. Brak automatycznej replikacji i nowych testów.

**Poprzedni pilot zakończony (6 listopada):** uczenie kwaternionowych współczynników kompensacji
podczas trzyepokowego douczania. Dwa warianty, jedna ostatnia konwolucja Q,
identyczny punkt startowy i maski. Parametry podczas treningu: 19 098 / 20 122;
po złożeniu obu sieci: **19 098**. Walidacja po naprawie: **56,7% / 56,9%**,
cross-entropy praktycznie bez zmiany (minimalnie gorsze w wariancie uczonym).
To jeden pilot, bez przekonującego dowodu korzyści. Stan: [state.json](results/trainable_maps_pilot/state.json),
wyniki: [REPORT.md](results/trainable_maps_pilot/REPORT.md).
Testy gradientów, złożenia i wznowienia oraz niezależna kontrola CPU przeszły.
Szczegółowy przegląd QNN i ograniczenia
nowości zapisano w [LITERATURE.md](LITERATURE.md).
Nic więcej nie jest automatycznie uruchamiane.

**Poprzedni pilot zakończony:** kompensacja przez końcowe wyniki klasyfikatora,
jedna późna konwolucja Q, CIFAR seed2026, 50% grup usuniętych w każdym miejscu.
Przed trzyepokową naprawą: **36,8%** wobec **34,9%** w kontroli lokalnej
z taką samą liczbą dodatkowych kroków. Po naprawie: **56,6% wobec 56,8%**.
Kwaternionowa struktura sześciu konwolucji i 19 098 parametrów zachowane.
To jeden pilot, bez wykazanej korzyści po douczeniu.
[Raport i niezależna kontrola CPU](results/tail_compensation_pilot/REPORT.md).
To badanie jest zamknięte. Starsze opisy celu skrótów poniżej są historyczne.

**6 listopada — replikacja direct-Q zakończona:** 14 przypadków, w tym
13 nowych trzyepokowych napraw i ponownie użyty wynik pilota. Przy tych samych
maskach i liczbie parametrów średnia poprawa względem wcześniejszej kompensacji
po naprawie wynosi na CIFAR **+1,27/+3,33 pp**, na EuroSAT **+0,23/+1,37 pp**
dla usunięcia 25%/50% grup w każdym miejscu. BN: trzy seedy; średnia CIFAR 50%
obejmuje pilot, a dwa dodatkowe seedy dały +4,2 i +3,3 pp. EuroSAT 50% ma też
jeden wynik −0,2 pp. Bez BN: tylko jeden seed, odpowiednio −0,1/+4,3 pp.
[Raport, wykres i kontrola wyników](results/direct_consumer_replication/REPORT.md).
Wyniki dotyczą walidacji, bez nowej oceny testu ani nowych modeli bazowych.
Kontrola 14 zapisanych modeli i 70 regresji przeszła; nie odtwarza ponownie
wszystkich predykcji walidacyjnych. Badanie zakończone, bez dalszych treningów
w kolejce. Oba skróty wznowienia prowadzą do tej zakończonej replikacji;
starsze opisy ich celu poniżej są historyczne.

[Osobna analiza zapisanych momentów](results/direct_consumer_pilot/gap_diagnostic/REPORT.md)
rozdziela ograniczenia wspólnych map kompensacji od swobody wag dostępnej
w tej samej warstwie Q. Dotyczy pięciu miejsc jednego pilota i lokalnej
funkcji celu; nie jest rozkładem accuracy ani dowodem nieusuwalnego kosztu QNN.
Dokładniejszy przegląd „Beyond Real Weights” i uwagi do kodu:
[LITERATURE.md](LITERATURE.md). Nie deklarujemy nowej metody LS ani pierwszeństwa.

**6 listopada — bieżący eksperyment:** wariant QNN bez BN zakończył 50 epok
(76,6% walidacji według zamrożonego wyboru CE; odpowiadający model z BN 78,3%,
jeden seed). Zakończono porównanie kompensacji kwaternionowej i swobodnej przy
kompresji wszystkich sześciu miejsc, z fizycznymi modelami i kontrolą swobody
wag podczas douczania: [raport](results/structure_cost_study/REPORT.md),
stan `results/structure_cost_study/state.json`, log `results/structure_cost_study/run.log`.
Ten etap jest zakończony; jego wyniki pozostają w katalogu badania.
Swobodny model ma więcej parametrów niż QNN o tej samej szerokości;
nie traktujemy ich jako modeli o równym budżecie. Nowe badanie nie ocenia testu.

Aktualizacja pracy do Marcusa (5 listopada): [przegląd literatury i decyzja](LITERATURE.md#paper-a-decision-review-2026-11-05),
[kontrola samego BatchNorm](results/paper_a_bn/REPORT.md) oraz
[historia odbudowy jakości po 1–3 epokach](results/paper_a_evidence/RECOVERY.md)
są gotowe. Nie ma nowych długich treningów. Końcowa ocena 333 zamrożonych modeli
jest opisana w [raporcie testowym](results/paper_a_final/REPORT.md);
stan: `results/paper_a_final/state.json`. Ten etap używa wcześniej odłożonego testu
EuroSAT; starsze opisy „test jeszcze nietknięty” dotyczą wcześniejszych badań.
Wznowienie oceny: `.venv-cuda\Scripts\python.exe -m scripts.paper_a_final`.

Zakończona seria (`WZNOW_BADANIA.cmd`, już policzona):
[dostrajanie z równym budżetem na CIFAR](results/finetune_compression/REPORT.md) oraz drugi
zbiór danych, EuroSAT z 4 pasmami jako `q = NIR + R·i + G·j + B·k`
([bazowe sieci](results/eurosat_cnn), [kompresja i dostrajanie](results/eurosat_compression/REPORT.md)).

Wynik (CNN6, wyjście bloku 2, usunięte 8/16 jednostek, walidacja, 3 seedy). Usuniętą jednostkę
odtwarzamy ze wszystkich pozostałych, `q_v ≈ Σ q_k·a_k`; dzięki domkniętości mnożenia
kwaternionów składa się to dokładnie w następną warstwę kwaternionową.
- Bez dotrenowania: CIFAR 73,2% wobec 54,0% przy losowym usuwaniu (pełna sieć 77,9%);
  EuroSAT 94,3% wobec 74,9% (pełna sieć 96,7%). Koszt zachowania struktury względem
  swobodnego podstawienia: 0,4–1,2 pp.
- Po 3 epokach dostrajania wszystkie metody różnią się o mniej niż ~0,5–1 pp na obu zbiorach,
  więc przy tym stopniu kompresji przewaga znika.
- Swobodna kompensacja z wielu źródeł jest znana (GRAIL2026); badamy tylko jej wersję
  ograniczoną do bloków Hamiltona. Nie deklarujemy nowej metody.

Główne krzywe jakość–rozmiar przy kompresji wszystkich warstw, z kontrolą treningu od zera,
są zakończone na CIFAR i EuroSAT, razem z zagęszczeniem krzywych. `URUCHOM_KRZYWA.cmd` oraz
`WZNOW_BADANIA.cmd` wznawiają całą bieżącą kolejkę. Stan:
`results/compression_curve_refinement_runner.json`, logi: `results/compression_curve_run.log` i
`results/compression_curve_eurosat_run.log`. Specyfikacja w
[PLAN_IJCNN2027.md](docs/PLAN_IJCNN2027.md).

Zagęszczenie krzywej **37,5% i 62,5%** zakończono na CIFAR i EuroSAT,
z tymi samymi kontrolami, dotrenowaniem i treningiem od zera.
Definicja: `configs/compression_curve_queue.json`; stan: `results/compression_curve_queue.json`.
Oba skróty prowadzą do całej kolejki. Dodatkowe punkty są osobno zarejestrowanym rozszerzeniem
po obejrzeniu zgrubnej krzywej.

Zakończono też badanie aktywacji: split LeakyReLU (0,1) i sprzężona MagnitudeTanh,
porównane z obecnym ReLU. Pierwszy etap obejmie CIFAR, CNN6 i trzy seedy (sześć nowych treningów
po 50 epok), kompresję 25%/50% oraz jednakowe dotrenowanie. Stan:
`results/activation_sensitivity_queue.json`; konfiguracja: `configs/activation_sensitivity.json`.
Wyniki są w `results/activation_sensitivity/REPORT.md`. Oba skróty prowadzą do całej kolejki.

Po aktywacjach zakończono **naprawę po kompresji przy małej liczbie etykiet**: CIFAR, CNN6,
trzy seedy, te same usuwane kanały z kompensacją i bez niej, 0/5/20 etykiet na klasę dostępnych
**tylko podczas naprawy**. Pełny model był wcześniej uczony ze wszystkimi etykietami;
5/20 to główne warianty, a 0 to kontrola graniczna. Porównujemy naprawę nadzorowaną,
pseudoetykiety aktualnego modelu i kontrolę z pełnym zamrożonym nauczycielem (54 krótkie
przebiegi). Stan: `results/pseudolabel_repair_queue.json`, konfiguracja:
`configs/pseudolabel_repair.json`. Oba istniejące skróty wznawiają całą kolejkę.

Główna oś artykułu: **kompresja istniejącej rodziny QNN — zachowanie struktury, koszt i ograniczenia**.
Zestawienie policzonych wyników: [przegląd QNN](results/qnn_compression_review/REPORT.md).
Zakończono również kontrolę struktury: bloki cyrkulantowe 4×4 o dokładnie tych
samych szerokościach i liczbie parametrów co QNN (CIFAR, trzy seedy, kompresja 25%/50%).
Konfiguracja: `configs/structure_control.json`; stan: `results/structure_control_queue.json`.
Porównamy również stratę względem pełnego modelu każdej rodziny. To nie dowód wyjątkowości
kwaternionów ani nowa metoda; pseudoetykiety pozostają osobnym pytaniem pomocniczym.

**Zakończono właściwy pilot SSL od początku treningu**, oparty na kodzie z sąsiedniego
`FewshotLearning`: 50 etykiet/klasę, miękkie pseudoetykiety, QNN i kontrola cyrkulantowa,
jeden seed. Po dwóch nowych treningach porównujemy kompresję i 12 krótkich kontynuacji.
Kalibracja jest oddzielona od treningu, pula U ma 47 500 obrazów. Ten pilot różni się od
poprzedniej naprawy modelu wcześniej uczonego ze wszystkimi etykietami. Status:
`results/ssl_compression_pilot_runner.json`; log: `results/ssl_compression_pilot_run.log`;
config: `configs/ssl_compression_pilot.json`. Kompensacja poprawiła jakość bezpośrednio po
kompresji, ale po 1000 krokach dalszego SSL jej przewaga nad usuwaniem tych samych jednostek
prawie zniknęła. Jest to pilot jednego seedu, nie dowód równoważności ani nowej metody.
Tytuł i deklaracja dla Marcusa pozostają do ustalenia na podstawie wyników.

1. Give the entire folder to Codex or Claude Code.
2. Codex: reads `AGENTS.md`; Claude Code: reads `CLAUDE.md`. Both point to `PROJECT.md`.
3. `LITERATURE.md` is the annotated bibliography. `references.bib` contains BibTeX entries with matching keys.
4. First coding task: Phase 0 of `PROJECT.md` — verify quaternion training correctness before pruning.

**Important:** This is a *project proposal*, not a finished paper. Bibliography metadata/links were manually checked against publisher or proceedings pages on 2026-11-02. Descriptions are short interpretations, **not verbatim abstracts**. Lack of an exact-match paper is not proof of priority.

Przegląd reprezentacji danych u autorów KHNN:
[Kodowanie wejścia](docs/INPUT_ENCODING_REVIEW.md) — obrazy RGB, szeregi czasowe
i różnica między opisanym ARGB a kolejnością ABGR w notebooku.

Wcześniejszy etap (później powtórzony dla CNN8 i dwóch poziomów kompresji w
[results/internal_replication](results/internal_replication/REPORT.md)): [scalanie wewnątrz sieci kwaternionowej](results/internal_merging/REPORT.md).
W trzech seedach wymuszenie struktury kolejnej warstwy zwiększyło błąd jej
wyjścia o 26–35% względem swobodnego podstawienia 4×4. Przy jednakowych
66 954 parametrach bezpośrednie dopasowanie kwaternionowe daje 74,73%
walidacji, a dopasowanie 4×4 z późniejszą projekcją Frobeniusa 73,63%: różnice
+0,7 / +1,8 / +0,8 pp. Pełna sieć ma 77,93%, więc koszt kompresji pozostaje.
Jeden blok i poziom kompresji; bez treningu CNN i nowej oceny testowej.
Kontrola analityczna pokazuje, że projekcja ważona aktywacjami odtwarza
bezpośrednie dopasowanie Q. Przewaga nad projekcją nieważoną nie stanowi
więc nowego algorytmu; badamy mierzalny koszt zachowania struktury warstwy.
Bieżący etap: `URUCHOM_KRZYWA.cmd` (także przez `WZNOW_BADANIA.cmd`).

Poprzedni etap: [scalanie przed klasyfikatorem rzeczywistym](results/quaternion_merging/REPORT.md).
Trzy istniejące modele zmniejszono z 73 898 do 64 298 parametrów (12,99%).
Walidacja: pełna sieć 77,93%, scalanie kwaternionowe 76,63%, scalanie
rzeczywiste 4×4 76,67%. Kontrola z samą korektą średniego wkładu dla par
kwaternionowych osiągnęła 76,77%; przewagi trafności bardziej złożonego
scalania nad tą kontrolą nie potwierdzono. Strata CE poprawia się względem
kontroli, szczególnie dla pełnego przekształcenia rzeczywistego.
Bez dodatkowego treningu CNN i bez nowej oceny testowej.
Odtworzenie poprzedniego etapu: `.venv-cuda\Scripts\python.exe -m scripts.run_quaternion_merging`.

Zakończony eksperyment: [wpływ BatchNorm przy przejściu do stałych
grup](results/routing_bn_diagnostic/REPORT.md). Przy niezmienionych wagach przeliczenie
BN podniosło walidację stałego przypisania z 56,0% do 66,3%; po przeliczeniu
obu wariantów nadal pozostało 12,7 pp różnicy względem miękkiego przypisania.
Jeden seed, bez nowej oceny testowej. [Pełny raport](results/routing_bn_diagnostic/REPORT.md).
Odtworzenie zakończonego etapu: `.venv-cuda\Scripts\python.exe -m scripts.routing_bn_study`.

Zakończony pilotaż: [uczenie składu grup z dyskretnym przypisaniem już
podczas treningu](results/hard_forward_grouping/REPORT.md). Po 50 epokach i jednakowym
przeliczeniu BN: uczone grupy 77,8% walidacji / 77,3% kalibracji, losowa
kontrola 78,1% / 77,2%. Zmieniły się cztery z ośmiu czwórek; przy ustaleniu
grup nie było skoku predykcji. Przewagi jakości nie wykazano; CE jest gorsze,
a zaplanowany próg uruchomienia replikacji nie został osiągnięty.
[Raport](results/hard_forward_grouping/REPORT.md) i
[składy grup](results/hard_forward_grouping/group_membership.png).
Odtworzenie zakończonego etapu: `.venv-cuda\Scripts\python.exe -m scripts.run_hard_forward_grouping`.

## Uruchomienie Phase 0

Projekt ma teraz działający test infrastruktury: mały klasyfikator
kwaternionowy, sprawdzanie gradientów i aktualizacji wag oraz zapis aktywacji
i wykresów. Test używa **danych syntetycznych**, osobnych zbiorów treningowego,
kalibracyjnego i testowego, jednego ziarna losowego oraz CPU. Nie jest wynikiem
badawczym na MNIST ani dowodem skuteczności kompresji.

Sprawdzone środowisko: Python 3.10.11, TensorFlow 2.20.0, Keras 3.11.3,
NumPy 2.2.6, Windows x86-64. Z katalogu projektu, w PowerShell:

```powershell
# Przy pierwszej instalacji (środowisko .venv zostało już utworzone lokalnie):
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Testy i uruchomienie; aktywacja środowiska nie jest wymagana:
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.phase0
```

Na Linux/macOS użyj `.venv/bin/python` zamiast `.\.venv\Scripts\python.exe`.
Takie systemy nie były sprawdzane w tej sesji.

Konfiguracja: [configs/phase0.json](configs/phase0.json). Przykład zapisu do
osobnego katalogu, aby zachować wcześniejsze wyniki:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase0 --output-dir results/moja-proba
```

Domyślne uruchomienie aktualizuje pliki w `results/phase0/`:

- [FEASIBILITY.md](results/phase0/FEASIBILITY.md) — raport techniczny i ograniczenia.
- `upstream_smoke.json` — pomiary niezmienionej biblioteki HypercomplexKeras.
- `local_smoke.json`, `training.csv` — test lokalnej warstwy i przebieg uczenia.
- `activations.npz` — aktywacje przed i po uczeniu, układ `(128, 6, 4)`.
- `similarity_matrix.png`, `redundancy_histogram.png` — diagnostyka na kalibracji.
- `synthetic_data.npz`, `model.keras`, `config.json` — dane, model i konfiguracja.
- `environment.freeze.txt` — pełne wersje środowiska, w tym zależności pośrednie.
- `reproducibility.json`, `pytest.xml` — kontrola powtarzalności i wynik testów.

Do odtworzenia dokładnych wersji w nowym środowisku można zamiast
`requirements.txt` zainstalować `results/phase0/environment.freeze.txt`.

**Wykryty problem:** HypercomplexKeras 1.0.2 nie rejestruje kernela warstwy
Dense w Keras 3. Test wykazał 12 zamiast 36 parametrów uczonych i zerową
aktualizację kernela. Zgodnie z `PROJECT.md` dodano minimalną warstwę zastępczą
w [src/models/quaternion_dense.py](src/models/quaternion_dense.py), z wagami
tworzonymi przez `add_weight`. Oryginalny pakiet nie został zmodyfikowany.

Sam test upstream można powtórzyć poleceniem:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_upstream
```

W sprawdzonym środowisku zwraca kod **1**, ponieważ dokumentuje rzeczywisty
błąd upstream. Pełne `scripts.phase0` zwraca **0**, gdy lokalna implementacja
przechodzi kontrole; zachowuje też negatywny wynik upstream w osobnym raporcie.

## Mały eksperyment RGB

`scripts.rgb_pilot` uruchamia małe modele Dense na podzbiorze CIFAR-10:
5000 obrazów treningowych, 1000 walidacyjnych, 1000 kalibracyjnych oraz
10 000 oficjalnych testowych. Podziały są rozłączne. Statystyki normalizacji
pochodzą wyłącznie z wybranych danych treningowych. Pierwsze pobranie danych
wymaga połączenia z internetem; archiwum ma około 163 MB i jest weryfikowane
oficjalną sumą SHA-256. Cache znajduje się w `.cache/keras/`.

Jeden piksel jest jednym kwaternionem. Porównujemy `(0,R,G,B)`, wariant
`(0,B,G,R)` z notebooka autorów oraz kontrolę rzeczywistą korzystającą z RGB.
Model kwaternionowy ma 132 490 rzeczywistych parametrów uczonych, a kontrola
132 579, czyli o 0,0672% więcej. Liczba kanałów ukrytych i koszt obliczeń nie
są dopasowane. To pilotaż, bez przycinania jednostek i bez deklaracji przewagi.

```powershell
.\.venv\Scripts\python.exe -m scripts.download_cifar10
.\.venv\Scripts\python.exe -m scripts.rgb_pilot
```

Pobieranie domyślnie korzysta z mirrora TensorFlow; `--source official`
wybiera serwer autorów CIFAR. Oba źródła wymagają identycznej sumy SHA-256.
Pełny log pobierania: `.cache/cifar10-download.log`.

Konfiguracja [configs/rgb_pilot.json](configs/rgb_pilot.json) ustala trzy ziarna
losowe i 20 epok dla każdego z trzech wariantów. Wybór modelu opiera się na
stracie walidacyjnej. Kalibracja służy do analizy aktywacji; test jest oceniany
dopiero po zakończeniu wszystkich treningów.

Wyniki trafiają do `results/rgb_pilot/`: `REPORT.md`, `summary.json`,
`comparison.csv`, wykresy uczenia i podobieństwa oraz katalogi poszczególnych
treningów z modelami, aktywacjami i kontrolami gradientów. Protokół:
[PROTOCOL.md](results/rgb_pilot/PROTOCOL.md). Skrypt chroni ukończone wyniki
przed nadpisaniem; kolejną próbę zapisz do nowego katalogu:

```powershell
.\.venv\Scripts\python.exe -m scripts.rgb_pilot --output-dir results/rgb-powtorzenie
```

Wykonany pilotaż: [raport](results/rgb_pilot/REPORT.md),
[wyniki CSV](results/rgb_pilot/comparison.csv) i
[wykresy podobieństwa](results/rgb_pilot/redundancy_histograms.png).
Średnia trafność testowa dla trzech ziaren wyniosła 37,78% dla `(0,R,G,B)`,
37,25% dla `(0,B,G,R)` i 37,61% dla kontroli rzeczywistej. Ten mały eksperyment
nie potwierdza przewagi modelu kwaternionowego. Nie znaleziono par całych
jednostek o użytym centrowanym podobieństwie cosinusowym ≥0,9; wynik ten
nie rozstrzyga ogólnej hipotezy o redundancji.

Zapisane artefakty można sprawdzić bez ponownego treningu ani inferencji:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_rgb_pilot
```

Weryfikator przelicza metryki z zapisanych logitów, liczy parametry bezpośrednio
w plikach wag, sprawdza wybór epoki na walidacji, rozłączność podziałów i sumy
kontrolne. Wynik bieżącej kontroli: **89 sprawdzeń, wszystkie poprawne**,
szczegóły w `results/rgb_pilot/verification.json`.

## CNN z celem 80% na CIFAR-10

Kolejny etap używa CNN z sześcioma konwolucjami, BatchNorm, ReLU, poolingiem,
dropoutem i globalnym uśrednianiem. Uczenie obejmuje 48 000 obrazów;
wcześniejsze 1000 walidacyjnych i 1000 kalibracyjnych pozostają odłożone.
Treningowe przycięcia i odbicia nie zmieniają kolejności kanałów RGB.

Model kwaternionowy reprezentuje piksel jako `(0,R,G,B)` i ma 73 898
rzeczywistych parametrów uczonych; kontrola rzeczywista ma 72 954.
Parametry BatchNorm i jego nieuczone bufory są raportowane osobno.
**80% jest celem do sprawdzenia**, a nie obietnicą wynikającą z architektury.
Pierwszy przebieg wykorzystuje jedno ziarno, bez przycinania jednostek.

Na tym komputerze trening korzysta z RTX 3050 Ti poprzez osobne środowisko
PyTorch/CUDA. Wcześniejsze środowisko TensorFlow i wyniki pilotażu pozostają
dostępne. Instalacja GPU jest duża; dane korzystają z tego samego zweryfikowanego
archiwum CIFAR w `.cache/keras/datasets/`.

```powershell
# Jednorazowo (lokalne środowisko zostało już przygotowane):
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt

# Testy nowej implementacji i pomiar kosztu przed treningiem:
.\.venv-cuda\Scripts\python.exe -m pytest tests_torch -q
.\.venv-cuda\Scripts\python.exe -m scripts.cifar_cnn --benchmark

# Pełny trening i ewentualne wznowienie po przerwaniu:
.\.venv-cuda\Scripts\python.exe -m scripts.cifar_cnn
.\.venv-cuda\Scripts\python.exe -m scripts.cifar_cnn --resume
```

Konfiguracja: [configs/cifar_cnn.json](configs/cifar_cnn.json).
Protokół: [results/cifar_cnn/PROTOCOL.md](results/cifar_cnn/PROTOCOL.md).
Wyniki zapisują się w `results/cifar_cnn/`. `last.pt` przechowuje stan treningu
wraz z RNG i optymalizatorem; `model.pt` jest modelem wybranym według straty
walidacyjnej. Nowy, niezależny przebieg zapisuj przez `--output-dir` do osobnego
katalogu. Wynik testowy jest obliczany po ukończeniu obu treningów.

Po ukończeniu niezależna kontrola zapisanych metryk i wag:

```powershell
.\.venv-cuda\Scripts\python.exe -m scripts.verify_cifar_cnn
```

Testy warstw nowej CNN są w `tests_torch/`; uruchamia się je w `.venv-cuda`.
Testy wcześniejszej implementacji Keras w `tests/` nadal korzystają z `.venv`.

Wykonany trening CNN został wznowiony od epoki 12 i ukończony dla obu modeli
do epoki 50. Dla jednego ziarna trafność testowa wyniosła **79,00%** dla modelu
kwaternionowego i **76,24%** dla rzeczywistego. Cel co najmniej 80% nie został
osiągnięty w tym przebiegu; różnica między modelami nie stanowi dowodu ogólnej
przewagi kwaternionów. [Raport CNN](results/cifar_cnn/REPORT.md) zawiera
szczegóły i ograniczenia. Niezależna kontrola zapisów przeszła 35 sprawdzeń,
w tym ciągłość epok i zachowanie liczby prób aktualizacji po wznowieniu.

## Analiza bloków i kontrolowana kompresja

[Analiza wszystkich bloków](results/block_analysis/REPORT.md) obejmuje
kalibrację i pojedyncze ablacje na walidacji, bez ponownego treningu i testu.
W trzecim bloku 26/32 kwaternionów spełnia próg małego wpływu przy usuwaniu
pojedynczo; podobieństwo słabo przewiduje ten wpływ (Spearman 0,223).
To nie oznacza, że można je wszystkie usunąć jednocześnie.
Pierwotny raport liczył 24/32 wskutek błędu na granicy progu zmiennoprzecinkowego;
errata zachowuje ten wynik i dokumentuje korektę, bez zmiany pomiarów i wyboru bloku.

Kontynuacja zamraża wcześniejszy protokół dla ziaren 2027 i 2028, zachowując
wyniki 2026. [Protokół trzech ziaren](results/cnn_multiseed/PROTOCOL.md) i
[protokół kompresji](results/cnn_compression/PROTOCOL.md) opisują osobne etapy.
Wznowienie całej sekwencji po przerwaniu:

```powershell
.\.venv-cuda\Scripts\python.exe -m scripts.research_pipeline
```

Powyższe polecenie dotyczy zakończonego pilotażu trzech seedów i kompresji.
Stan epoki znajduje się w `results/cnn_multiseed/seed_*/{quaternion,real}/last.pt`.
Ukończone modele są ponownie wykorzystywane; źródła i konfiguracje są sprawdzane.

Testy ablacji oraz fizycznego usuwania kanałów są częścią `tests_torch`.
Porównanie usuwa 0/25/50% całych kwaternionów z drugiej konwolucji bloku 3,
przebudowując również BatchNorm i readout. Procent usuniętych jednostek bloku
nie jest procentem zaoszczędzonych parametrów całej sieci. Nie ma dotrenowania.

Etap został ukończony dla ziaren 2026, 2027 i 2028. Średnia trafność testowa:
**78,99% ± 0,22 pp** dla CNN kwaternionowej oraz **75,94% ± 0,26 pp**
dla rzeczywistej; odchylenie dotyczy trzech ziaren, nie dowodzi ogólnej przewagi.
[Raport trzech ziaren](results/cnn_multiseed/REPORT.md).

Fizyczne usunięcie 25% jednostek ostatniego bloku daje **12,99% mniej**
rzeczywistych parametrów całej sieci i około **77,5%** trafności. Użyte
podobieństwo aktywacji nie pokonało losowego usuwania: 77,48% wobec 77,50%.
Przy 50% usuniętych jednostek (25,98% oszczędności parametrów) wyniki to
74,12% i 74,08%. To negatywny wynik dla przewagi tego kryterium.
[Raport kompresji](results/cnn_compression/REPORT.md),
[wykres](results/cnn_compression/compression_curves.png),
[aktualny stan i następny krok](docs/RESEARCH_STATUS.md).
Kontrole: 22 testy implementacji, 35 sprawdzeń każdego nowego modelu bazowego
oraz 1285 sprawdzeń artefaktów kompresji; wszystkie poprawne.

## Sprawdzenie kryterium uwzględniającego BatchNorm

Osobny, zamrożony eksperyment używa normy efektywnych wag konwolucji po
uwzględnieniu skalowania przez BN, wspólnie dla czterech kanałów kwaternionu.
Wykorzystuje te same trzy modele i fizyczny rozmiar kompresji, bez uczenia.
Poprzednie wyniki pozostają zachowane.

Wynik: **76,10%** przy 25% usuniętych jednostek oraz **73,30%** przy 50%,
wobec losowych **77,50%** i **74,08%**. Nowe kryterium było gorsze od
średniego losowania w każdym z trzech ziaren przy obu rozmiarach.
Samo uwzględnienie BN nie dało lepszego wskazania ważności kanałów.
[Raport BN](results/bn_compression/REPORT.md),
[porównanie](results/bn_compression/comparison.png),
[zamrożony protokół](results/bn_compression/PROTOCOL.md).
Weryfikacja: **25 testów implementacji i 130 kontroli artefaktów**, poprawne.

```powershell
.\.venv-cuda\Scripts\python.exe -m scripts.bn_compression
.\.venv-cuda\Scripts\python.exe -m scripts.verify_bn_compression
.\.venv-cuda\Scripts\python.exe -m scripts.summarize_research
```

Pierwsza komenda wykorzystuje istniejące wyniki przy identycznym protokole;
nie uruchamiaj równoległych zapisów do tego samego katalogu. Niezależne
powtórzenie wymaga osobnego `--output-dir`.

## Całe kwaterniony a niezależne korelacje składowych

Porównanie na zapisanej kalibracji i pojedynczych ablacjach walidacyjnych
zostało wykonane bez treningu, inferencji i użycia testu. W głównym bloku 3
zwykłe korelacje skalarne lepiej przewidują skutki usunięcia całej jednostki
dla ziaren 2026 i 2027; grupowanie wygrywa dla 2028. Średnia różnica
Spearmana (grupowanie minus korelacje) wynosi -0,052, SD 0,212 dla trzech
ziaren. Brak powtarzalnej przewagi badanego grupowania.

Oba rankingi wskazują jednostki o małym wpływie przy usuwaniu pojedynczo.
Ten wynik nie oznacza bezpiecznego jednoczesnego usunięcia wybranej ćwiartki.
Wariant skalarny wyklucza dopasowania do wszystkich czterech własnych
składowych, bo one znikają razem. Wyniki wcześniejszego przycinania nie są
przepisywane na te nowe rankingi diagnostyczne.

[Raport porównania](results/group_scalar/REPORT.md),
[wykres głównego bloku](results/group_scalar/primary_comparison.png),
[protokół](results/group_scalar/PROTOCOL.md),
[hipoteza o strukturze parametrów](docs/STRUCTURE_HYPOTHESIS.md).
Kontrole: **28 testów implementacji i 200 kontroli artefaktów**, poprawne.
Zachowano też zapis pierwszej błędnej kontroli kształtu i jej korekty.

```powershell
.\.venv-cuda\Scripts\python.exe -m scripts.verify_group_scalar
# Niezależne powtórzenie analizy:
.\.venv-cuda\Scripts\python.exe -m scripts.group_scalar_diagnostics --output-dir results/group_scalar-repro
```

## Uczenie przynależności kanałów do grup

Moduł pomiędzy blokami 1 i 2 dobiera przypisanie 32 kanałów do pozycji
ośmiu czwórek. Podczas uczenia używa miękkiej macierzy Sinkhorna, potem
eksportuje ją do stałej permutacji. Nie zmienia reguły mnożenia kwaternionów.
[Protokół eksperymentu](results/learned_grouping_pilot/PROTOCOL.md),
[wstępna ocena nowatorskości](docs/LEARNED_GROUPING_NOVELTY_AUDIT.md).

Prototyp przeszedł kontrolę gradientów i eksportu na GPU. Pilotaż dla jednego
ziarna porównuje wcześniejszy równoważny podział naturalny z podziałem losowym
i uczonym. Wszystkie modele docelowe mają 73 898 uczonych parametrów; podczas
miękkiego uczenia dochodzą 1024 zmienne i koszt mieszania kanałów.
Nie jest to twierdzenie o identycznym budżecie treningowym ani o kompresji.

```powershell
# Uruchomienie lub wznowienie całej sekwencji; tylko jedna kopia naraz:
.\.venv-cuda\Scripts\python.exe -m scripts.run_learned_grouping_study
```

Stan optymalizatora, generatorów losowych i fazy miękkiej/twardej
znajduje się w `last.pt`.
Aktywny poprawiony przebieg trafia do `results/learned_grouping_pilot/`.
Przerwany sześcioepokowy przebieg techniczny jest zachowany osobno w
`results/learned_grouping/` i nie wchodzi do porównania.
Warunkowa kompresja uruchamia się tylko po przejściu wcześniej zapisanego
progu walidacyjnego; nie zakłada sukcesu ani nowatorskości metody.

Pilotaż zakończony, weryfikacja po wznowieniu 2026-11-03: **79,00%** testowo
dla wcześniejszego podziału, **78,14%** dla stałego losowego i **77,61%**
dla uczonego po zamrożeniu. Jedno ziarno, 2026. Zmieniły się dwie z ośmiu
czwórek, ale jakość nie wzrosła. Przy samym zamrożeniu walidacja spadła
z **79,20% do 56,00%**; późniejsze uczenie odzyskało część jakości.
Warunek dalszej kompresji nie został spełniony i ten etap pominięto.
[Raport pilotażu](results/learned_grouping_pilot/REPORT.md),
[krzywe uczenia](results/learned_grouping_pilot/learning_curves.png),
[miękkie i twarde przypisania](results/learned_grouping_pilot/routing_matrices.png).
Kontrole zakończenia: 64 sprawdzenia pilotażu i 7 sprawdzeń decyzji o
nieuruchamianiu kompresji — poprawne. To negatywny wynik tej konfiguracji,
bez deklaracji nowej skutecznej metody.
