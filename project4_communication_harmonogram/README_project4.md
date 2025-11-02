# Space Communication Windows Scheduler

## Opis projektu

Projekt służy do przewidywania **okien komunikacyjnych dla satelitów** względem stacji naziemnych.  
Dzięki bibliotece [Skyfield](https://rhodesmill.org/skyfield/) można dokładnie obliczyć pozycję satelity na podstawie TLE (Two-Line Elements) i określić, kiedy satelita będzie widoczny z wybranych lokalizacji na Ziemi.

Aplikacja generuje:

- Harmonogram przelotów satelity dla wybranych stacji naziemnych.
- Wykresy Gantta pokazujące czas, w którym możliwa jest komunikacja lub obserwacja satelity.
- Informacje o tym, czy satelita jest widoczny gołym okiem i czy stacja znajduje się w nocy.

---

## Wymagania

- Python 3.10+
- Biblioteki Python:
  ```bash
  pip install skyfield numpy matplotlib requests
  ```


- Dostęp do internetu (do pobierania aktualnych TLE satelitów z CelesTrak).

---

## Parametry konfiguracyjne

- `SAT_NAME` – nazwa satelity (np. `"ISS (ZARYA)"`)
- `SITE_TLE` – URL do pliku TLE, np. [CelesTrak stations.txt](https://celestrak.org/NORAD/elements/stations.txt)
- `GROUND_STATIONS` – lista stacji naziemnych w formacie:

  ```python
  {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122, "min_elev": 20}
  ```

- `REFRESH` – czas odświeżania harmonogramu w godzinach
- `PREDICT_TIME` – liczba godzin do przodu, na które przewidujemy przeloty
- `DEGREE` – minimalny kąt nad horyzontem dla okna komunikacji
- `ONLY_STATION_NIGHT` – True jeśli interesują nas przeloty tylko w nocy dla stacji
- `ONLY_VISIBLE` – True jeśli interesują nas przeloty tylko gdy satelita jest oświetlony przez Słońce

---

## Funkcje

### `load_data()`

- Pobiera aktualne TLE dla satelitów.
- Zwraca obiekt `EarthSatellite` dla wybranego satelity.
- W przypadku braku satelity w pliku TLE wyświetla błąd i kończy program.

### `is_visible(satellite, t)`

- Sprawdza, czy satelita jest widoczny gołym okiem w danym momencie.
- Uwzględnia, czy satelita nie znajduje się w cieniu Ziemi.
- Wykorzystuje pozycję satelity i Słońca w układzie ECI, licząc, czy linia Słońce → satelita przecina Ziemię.

### `is_station_in_night(station_topos, t)`

- Sprawdza, czy stacja naziemna znajduje się w nocy cywilnej (Słońce poniżej -6°).
- Używa pozycji Słońca względem obserwatora na powierzchni Ziemi.

### `compute_windows(satellite, ground_station, hours)`

- Oblicza okna komunikacyjne dla satelity względem stacji.
- Wykorzystuje `satellite.find_events` do określenia wschodu, kulminacji i zachodu satelity.
- Zwraca listę słowników zawierających:

  - `start` – czas rozpoczęcia okna
  - `end` – czas zakończenia okna
  - `duration` – czas trwania okna w sekundach

### `visualization(windows)`

- Wyświetla harmonogram w terminalu w formie tekstowej.
- Pokazuje start, koniec i nazwę stacji.

### `Gantt_chart(windows)`

- Tworzy wykres Gantta w matplotlib.
- Kolory pasków:

  - zielony – satelita widoczny + stacja w nocy
  - żółty – stacja w nocy, satelita może być w cieniu
  - szary – satelita niewidoczny

- Oś czasu z podziałką godzinową i minutową.
- Wszystkie napisy i osie w kolorze białym na ciemnym tle.
- Możliwe rozszerzenie kolorów przy dodaniu kolejnych stacji i warunków widoczności.

---

## Przykładowe stacje naziemne

- Europejskie: Warsaw, Berlin, London, Paris
- Obserwatoria na innych kontynentach: Mauna Kea (Hawaje, USA), Siding Spring (Australia)

---

## Uruchamianie programu

```bash
python sat_communication_schedule.py
```

Program działa w pętli:

1. Pobiera aktualne TLE.
2. Oblicza okna komunikacyjne dla wszystkich stacji.
3. Wyświetla harmonogram w terminalu.
4. Rysuje wykres Gantta.
5. Czeka określoną liczbę godzin (`REFRESH`) i powtarza procedurę.

---

## Uwagi

- Program działa w czasie UTC.
- Wykresy mogą pokazywać różne kolory tylko jeśli stacje spełniają różne warunki widoczności i nocy.
- Możliwość rozszerzenia:

  - Eksport harmonogramu do CSV/JSON.
  - Powiadomienia o nadchodzących przelotach.
  - Wykresy w czasie rzeczywistym.

---

## Kolory pasków w Gantcie

| Kolor             | Znaczenie                                                      |
| ----------------- | -------------------------------------------------------------- |
| Zielony (#00FF00) | Satelita widoczny + stacja w nocy (pełna widoczność)           |
| Żółty (#FFD700)   | Satelita może być w cieniu, ale stacja w nocy                  |
| Szary (#A9A9A9)   | Satelita niewidoczny, stacja w ciągu dnia lub brak widoczności |

