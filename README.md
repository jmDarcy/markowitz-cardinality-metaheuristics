# Markowitz Portfolio Optimization with Cardinality Constraint

Projekt dotyczy minimalizacji wariancji portfela Markowitza z dodatkowym ograniczeniem kardynalnosci, czyli limitem liczby aktywow o niezerowej wadze. Poniewaz takie ograniczenie wprowadza komponent dyskretny, klasyczny problem portfelowy przestaje byc prostym zadaniem wypuklym. W projekcie porownano trzy metaheurystyki: Simulated Annealing, Genetic Algorithm oraz Particle Swarm Optimization.

## Cel projektu

Celem jest przygotowanie kompletnego procesu obliczeniowego, ktory:

- generuje syntetyczne szeregi stop zwrotu,
- estymuje oczekiwane stopy zwrotu i macierz kowariancji,
- rozwiazuje problem minimalizacji wariancji z ograniczeniem liczby aktywow,
- porownuje jakosc, stabilnosc i czas dzialania algorytmow,
- zapisuje wyniki w postaci tabel, wykresow oraz raportu LaTeX.

## Rozwazany problem

Dla wektora wag `w`, oczekiwanych stop zwrotu `mu` i macierzy kowariancji `Sigma` minimalizowana jest wariancja portfela:

```text
min w' Sigma w
```

przy ograniczeniach:

- suma wag rowna 1,
- brak krotkiej sprzedazy, czyli `w_i >= 0`,
- minimalna oczekiwana stopa zwrotu,
- maksymalnie `A` aktywow w portfelu.

Eksperyment przeprowadzono dla wartosci `A = {2, 3, 5, 10, 15}`.

## Zawartosc repozytorium

```text
.
|-- notebooks/
|   `-- final_notebook.ipynb          # finalny notebook z implementacja i wynikami
|-- data/
|   `-- synthetic_returns.csv         # kanoniczny syntetyczny zbior danych
|-- charts/                           # wykresy wynikowe PNG
|-- tables/                           # tabele wynikowe CSV i TEX
|-- latex/
|   |-- main.tex                      # zrodlo raportu
|   |-- main.pdf                      # skompilowany raport koncowy
|   |-- figures/                      # figury uzywane przez raport
|   |-- tables/                       # tabele uzywane przez raport
|   `-- references.bib
|-- logs/
|   |-- experiment_config.json        # parametry eksperymentu
|   `-- validation_summary.txt
|-- _build_latex_project.py           # skrypt generujacy wyniki i raport
|-- .gitignore
`-- README.md
```

Repozytorium celowo pomija pliki robocze i duplikaty: katalogi `data/raw/`, `data/synthetic/`, artefakty kompilacji LaTeX (`*.aux`, `*.log`, `*.toc`, `*.out`), pliki Excel oraz zdublowane PDF-y z katalogu `reports/compiled/`.

## Najwazniejsze wyniki

Projekt porownuje algorytmy pod wzgledem:

- minimalnej wariancji portfela,
- zmiennosci i oczekiwanej stopy zwrotu,
- stabilnosci wynikow w powtorzeniach,
- czasu dzialania,
- efektywnej liczby aktywow i koncentracji portfela,
- spelnienia ograniczen formalnych.

W przeprowadzonym eksperymencie najlepszy wynik wariancyjny uzyskano dla algorytmu PSO przy `A = 10`. Interpretacja wyniku powinna jednak uwzgledniac rowniez czas obliczen, stabilnosc oraz poziom dywersyfikacji.

## Uruchomienie

Wymagane biblioteki:

```bash
pip install numpy pandas matplotlib openpyxl
```

Uruchomienie notebooka:

```bash
jupyter notebook notebooks/final_notebook.ipynb
```

Odtworzenie wynikow i raportu z poziomu skryptu:

```bash
python _build_latex_project.py
```

Kompilacja raportu LaTeX:

```bash
cd latex
pdflatex main.tex
pdflatex main.tex
```

Gotowy raport znajduje sie w `latex/main.pdf`.

## Uwagi

Dane sa syntetyczne i sluza do demonstracji metod optymalizacji, a nie do budowy realnej rekomendacji inwestycyjnej. Wyniki zaleza od przyjetych parametrow symulacji, liczby powtorzen oraz sposobu naprawiania portfeli naruszajacych ograniczenia.
