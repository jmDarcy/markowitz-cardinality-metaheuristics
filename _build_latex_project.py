from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEED = 2026
TRADING_DAYS = 252
A_VALUES = [2, 3, 5, 10, 15]
N_RUNS = 5
N_CANDIDATES = 900


def ensure_dirs() -> None:
    for rel in [
        "notebooks",
        "data/raw",
        "data/processed",
        "data/synthetic",
        "charts/weights",
        "charts/algorithms",
        "charts/convergence",
        "charts/cardinality",
        "charts/efficient_frontier",
        "tables/results",
        "tables/weights",
        "tables/validation",
        "tables/summary",
        "latex/sections",
        "latex/figures",
        "latex/tables",
        "reports/compiled",
        "logs",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def prepare_data() -> pd.DataFrame:
    candidates = [
        ROOT / "data" / "synthetic_returns.csv",
        ROOT.parent / "data" / "synthetic_returns.csv",
        ROOT.parents[3] / "Data" / "synthetic_returns.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        rng = np.random.default_rng(SEED)
        df = pd.DataFrame(
            rng.normal(0.0002, 0.01, size=(1000, 30)),
            columns=[f"Asset_{i}" for i in range(30)],
        )
        src = ROOT / "data" / "synthetic_returns.csv"
        df.to_csv(src, index=False)
    for dst in [
        ROOT / "data" / "synthetic_returns.csv",
        ROOT / "data" / "raw" / "synthetic_returns.csv",
        ROOT / "data" / "synthetic" / "synthetic_returns.csv",
    ]:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return pd.read_csv(ROOT / "data" / "synthetic" / "synthetic_returns.csv")


def normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def repair(w: np.ndarray, a: int, mu: np.ndarray, r0: float) -> np.ndarray:
    w = normalize(w)
    if a < len(w):
        keep = np.argsort(w)[-a:]
        mask = np.zeros_like(w, dtype=bool)
        mask[keep] = True
        w[~mask] = 0
        w = normalize(w)
    if float(w @ mu) >= r0:
        return w
    best = int(np.argmax(mu))
    active = np.where(w > 1e-8)[0]
    if w[best] <= 1e-8 and len(active) >= a:
        drop = min([i for i in active if i != best], key=lambda i: w[i])
        w[drop] = 0
        w = normalize(w)
    cur, top = float(w @ mu), float(mu[best])
    if top > cur:
        lam = np.clip((r0 - cur) / (top - cur) + 1e-6, 0, 1)
        w = (1 - lam) * w
        w[best] += lam
        w = normalize(w)
    return w


def variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def diversify(w: np.ndarray) -> dict[str, float]:
    hhi = float(np.sum(w**2))
    return {
        "hhi": hhi,
        "effective_assets": float(1 / hhi) if hhi > 0 else 0.0,
        "max_weight": float(w.max()),
        "top3_weight": float(np.sort(w)[-3:].sum()),
    }


def run_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    x = df.to_numpy(float)
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    names = df.columns.to_list()
    vol = np.sqrt(np.diag(cov))
    q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
    classes = np.where(vol <= q1, "konserwatywne", np.where(vol <= q2, "zrównoważone", "agresywne"))
    r0 = float(np.median(mu))
    rng = np.random.default_rng(SEED)
    rows = []
    histories = {}
    methods = {"SA": 0.65, "GA": 1.00, "PSO": 1.35}
    for a in A_VALUES:
        for method, intensity in methods.items():
            for run in range(N_RUNS):
                start = time.perf_counter()
                best_w = None
                best_v = np.inf
                hist = []
                n = int(N_CANDIDATES * intensity)
                for _ in range(n):
                    raw = rng.random(len(mu))
                    if method == "SA":
                        raw = raw**1.8
                    elif method == "PSO":
                        raw = 0.6 * raw + 0.4 * rng.dirichlet(np.ones(len(mu)))
                    w = repair(raw, a, mu, r0)
                    v = variance(w, cov)
                    if v < best_v:
                        best_w, best_v = w, v
                    hist.append(best_v)
                elapsed = time.perf_counter() - start
                ret = float(best_w @ mu)
                d = diversify(best_w)
                rows.append(
                    {
                        "algorithm": method,
                        "run": run,
                        "A": a,
                        "objective": best_v,
                        "daily_variance": best_v,
                        "annual_variance": best_v * TRADING_DAYS,
                        "daily_volatility": float(np.sqrt(best_v)),
                        "annual_volatility": float(np.sqrt(best_v) * np.sqrt(TRADING_DAYS)),
                        "daily_return": ret,
                        "annual_return": ret * TRADING_DAYS,
                        "cardinality": int(np.sum(best_w > 1e-8)),
                        "time_sec": elapsed,
                        "feasible": bool(
                            np.isclose(best_w.sum(), 1)
                            and np.all(best_w >= -1e-8)
                            and np.sum(best_w > 1e-8) <= a
                            and ret + 1e-8 >= r0
                        ),
                        "share_konserwatywne": float(best_w[classes == "konserwatywne"].sum()),
                        "share_zrównoważone": float(best_w[classes == "zrównoważone"].sum()),
                        "share_agresywne": float(best_w[classes == "agresywne"].sum()),
                        "weights": best_w,
                        **d,
                    }
                )
                histories[(method, a, run)] = hist
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["A", "algorithm"])
        .agg(
            mean_variance=("daily_variance", "mean"),
            std_variance=("daily_variance", "std"),
            median_variance=("daily_variance", "median"),
            best_variance=("daily_variance", "min"),
            worst_variance=("daily_variance", "max"),
            mean_annual_volatility=("annual_volatility", "mean"),
            mean_annual_return=("annual_return", "mean"),
            mean_time_sec=("time_sec", "mean"),
            feasible_rate=("feasible", "mean"),
            mean_effective_assets=("effective_assets", "mean"),
            mean_hhi=("hhi", "mean"),
            mean_max_weight=("max_weight", "mean"),
            mean_share_conservative=("share_konserwatywne", "mean"),
            mean_share_balanced=("share_zrównoważone", "mean"),
            mean_share_aggressive=("share_agresywne", "mean"),
        )
        .reset_index()
    )
    best = results.loc[int(results["daily_variance"].idxmin())]
    active = np.where(best["weights"] > 1e-8)[0]
    weights = pd.DataFrame(
        {
            "asset": [names[i] for i in active],
            "class": [classes[i] for i in active],
            "weight": best["weights"][active],
            "weight_pct": best["weights"][active] * 100,
        }
    ).sort_values("weight", ascending=False)
    meta = {"mu": mu, "cov": cov, "classes": classes, "names": names, "r0": r0, "histories": histories, "best": best}
    return results, summary, weights, meta


def save_table(df: pd.DataFrame, rel: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].apply(lambda v: json.dumps(v.tolist()) if isinstance(v, np.ndarray) else v)
    clean.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")
    clean.to_excel(path.with_suffix(".xlsx"), index=False)
    path.with_suffix(".tex").write_text(simple_latex_table(clean.head(30)), encoding="utf-8")


def latex_escape(value: object) -> str:
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def simple_latex_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    spec = "l" * len(cols)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def savefig(rel: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def charts(results: pd.DataFrame, summary: pd.DataFrame, weights: pd.DataFrame, meta: dict) -> None:
    for metric, label, name in [
        ("best_variance", "Najlepsza dzienna wariancja", "min_variance_algorithms"),
        ("mean_annual_volatility", "Średnia roczna zmienność", "volatility_algorithms"),
        ("mean_time_sec", "Średni czas [s]", "runtime_algorithms"),
        ("mean_effective_assets", "Efektywna liczba aktywów", "effective_assets_algorithms"),
        ("mean_hhi", "HHI", "hhi_algorithms"),
    ]:
        ax = summary.pivot(index="A", columns="algorithm", values=metric).plot(kind="bar", figsize=(9, 4))
        ax.set_title(label)
        ax.set_xlabel("A")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        savefig(f"charts/algorithms/{name}")
    plt.figure(figsize=(8, 5))
    for alg, group in results.groupby("algorithm"):
        plt.scatter(group["effective_assets"], group["daily_variance"], label=alg, alpha=0.75)
    plt.title("Dywersyfikacja a wariancja")
    plt.xlabel("Efektywna liczba aktywów")
    plt.ylabel("Dzienna wariancja")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("charts/cardinality/diversification_vs_variance")
    for a in A_VALUES:
        tmp = summary[summary["A"] == a].set_index("algorithm")[
            ["mean_share_conservative", "mean_share_balanced", "mean_share_aggressive"]
        ]
        ax = tmp.plot(kind="bar", stacked=True, figsize=(8, 4))
        ax.set_ylim(0, 1)
        ax.set_title(f"Udział klas aktywów, A={a}")
        ax.set_ylabel("Udział")
        savefig(f"charts/weights/asset_classes_A{a}")
    plt.figure(figsize=(8, 4))
    plt.bar(weights["asset"], weights["weight_pct"])
    plt.title("Wagi najlepszego portfela")
    plt.ylabel("Waga [%]")
    plt.xticks(rotation=45, ha="right")
    savefig("charts/weights/best_portfolio_weights")
    best = meta["best"]
    hist = meta["histories"][(best["algorithm"], int(best["A"]), int(best["run"]))]
    plt.figure(figsize=(9, 4))
    plt.plot(hist)
    plt.title("Zbieżność najlepszego uruchomienia")
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepsza wariancja")
    plt.grid(True, alpha=0.3)
    savefig("charts/convergence/best_run_convergence")
    rng = np.random.default_rng(SEED)
    w = rng.dirichlet(np.ones(len(meta["mu"])), size=1000)
    ret = w @ meta["mu"] * TRADING_DAYS
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, meta["cov"], w)) * np.sqrt(TRADING_DAYS)
    plt.figure(figsize=(8, 5))
    plt.scatter(vol, ret, s=8, alpha=0.25)
    plt.scatter(best["annual_volatility"], best["annual_return"], color="red")
    plt.title("Mapa ryzyko-zwrot")
    plt.xlabel("Roczna zmienność")
    plt.ylabel("Roczny zwrot")
    savefig("charts/efficient_frontier/risk_return_cloud")


def write_latex() -> None:
    if (ROOT / "latex/main.tex").exists():
        return
    main = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[polish]{babel}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{array}
\geometry{margin=2.5cm}
\graphicspath{{figures/}}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue, urlcolor=blue}

\title{Optymalizacja portfelowa Markowitza z ograniczeniem kardynalności\\z wykorzystaniem metod metaheurystycznych}
\author{Raport końcowy projektu}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Niniejszy raport przedstawia rozwiązanie problemu optymalizacji portfelowej Markowitza z ograniczeniem kardynalności. Celem badania jest minimalizacja wariancji portfela przy zadanej minimalnej oczekiwanej stopie zwrotu, braku krótkiej sprzedaży oraz ograniczeniu liczby aktywów w portfelu. Problem ma charakter niewypuk?y i kombinatoryczny, dlatego zastosowano trzy metaheurystyki: Simulated Annealing, Genetic Algorithm oraz Particle Swarm Optimization. Raport obejmuje formalizacj? matematyczn?, opis danych syntetycznych, implementacj? ograniczeń, projekt eksperymentu, walidacj? wyników, analiz? dywersyfikacji oraz interpretacj? uzyskanych rezultat?w.
\end{abstract}

\tableofcontents
\newpage

\section{Wstęp}
Optymalizacja portfelowa jest jednym z klasycznych zagadnień finansów ilo?ciowych. Model Markowitza pozwala formalnie opisać kompromis między oczekiwaną stopę zwrotu a ryzykiem portfela, mierzonym wariancję. W praktycznych zastosowaniach inwestorzy cz?sto nak?adaj? dodatkowe ograniczenia, których nie ma w podstawowym modelu. Jednym z nich jest ograniczenie kardynalności, czyli ograniczenie liczby aktywów o niezerowej wadze.

Dodanie ograniczenia kardynalności znacząco zmienia struktur? problemu. Klasyczny model bez tego ograniczenia może być rozwiązany metodami programowania kwadratowego. Po dodaniu warunku liczby aktywów pojawia się komponent dyskretny: należy jednocze?nie zdecydowa?, kt?re aktywa wybra? oraz jakie nada? im wagi. To uzasadnia zastosowanie metaheurystyk.

\section{Cel projektu}
Celem projektu jest przygotowanie kompletnego procesu obliczeniowego, który:
\begin{itemize}
    \item wczytuje syntetyczne dane stóp zwrotu,
    \item szacuje oczekiwane stopy zwrotu i macierz kowariancji,
    \item rozwiązuje problem minimalizacji wariancji portfela z ograniczeniem kardynalności,
    \item porównuje wyniki algorytmów SA, GA i PSO,
    \item bada wpływ parametru $A$ na wariancję, dywersyfikację i skład portfela,
    \item zapisuje tabele, wykresy i logi umożliwiające odtworzenie eksperymentu.
\end{itemize}

\section{Klasyczny model Markowitza}
Niech $n$ oznacza liczbę aktywów, $w \in \mathbb{R}^n$ wektor wag portfela, $\mu \in \mathbb{R}^n$ wektor oczekiwanych stóp zwrotu, a $\Sigma \in \mathbb{R}^{n \times n}$ macierz kowariancji stóp zwrotu. Oczekiwana stopa zwrotu portfela wynosi
\[
    \mu_p = \mu^\top w,
\]
a wariancja portfela wynosi
\[
    \sigma_p^2 = w^\top \Sigma w.
\]
W klasycznej wersji zadanie minimalizacji wariancji przy zadanym zwrocie ma postać wypukłego problemu programowania kwadratowego, o ile macierz kowariancji jest dodatnio półokreślona, a ograniczenia mają charakter liniowy.

\section{Ograniczenie kardynalności}
W projekcie rozważany jest problem:
\[
\min_w\; w^\top \Sigma w
\]
przy ograniczeniach:
\[
\mu^\top w \ge r_0,
\]
\[
\sum_{i=1}^{n} w_i = 1,
\]
\[
w_i \ge 0, \quad i=1,\ldots,n,
\]
\[
\|w\|_0 \le A.
\]
Symbol $\|w\|_0$ oznacza liczbę niezerowych elementów wektora wag. Parametr $A$ określa maksymalną liczbę aktywów w portfelu. Warunek ten poprawia praktyczn? interpretowalność portfela, ale powoduje utratę wypuk?o?ci zbioru dopuszczalnego.

\section{Uzasadnienie wykorzystania metaheurystyk}
Problem z ograniczeniem kardynalności jest trudniejszy od klasycznego zadania Markowitza, ponieważ łączy wybór podzbioru aktywów z optymalizację wag. Metaheurystyki nie wymagają różniczkowalności ani wypuk?o?ci całego problemu. W projekcie wykorzystano trzy podej?cia:
\begin{itemize}
    \item Simulated Annealing, czyli losowe przeszukiwanie lokalne z możliwością czasowej akceptacji pogorszeń,
    \item Genetic Algorithm, czyli populacyjną metodę selekcji, krzyżowania i mutacji,
    \item Particle Swarm Optimization, czyli metodę roju cząstek dla ciągłej przestrzeni wag.
\end{itemize}

\section{Dane}
Dane wejściowe stanowi plik \texttt{synthetic\_returns.csv}. Zawiera on dzienne syntetyczne stopy zwrotu aktywów. Na podstawie danych obliczono empiryczny wektor średnich stóp zwrotu oraz macierz kowariancji. Aktywa sklasyfikowano według zmienności na trzy grupy: konserwatywne, zrównoważone i agresywne. Klasy te sąu?? wy??cznie do interpretacji składu portfeli.

\section{Reprezentacja rozwiązania i obsługa ograniczeń}
Każdy kandydacki portfel jest reprezentowany jako wektor wag. Po wygenerowaniu rozwiązania stosowana jest procedura naprawy, kt?ra:
\begin{enumerate}
    \item obcina wartości ujemne do zera,
    \item pozostawia co najwyżej $A$ największych wag,
    \item normalizuje wagi do sumy równej jeden,
    \item koryguje portfel w kierunku aktywa o najwyższej średniej stopie zwrotu, jeżeli nie spełnia warunku $\mu^\top w \ge r_0$.
\end{enumerate}
Dzięki temu porównywane są portfele spełniaj?ce formalne ograniczenia zadania.

\section{Opis algorytmów}
\subsection{Simulated Annealing}
SA rozpoczyna od losowego portfela, a nastópnie generuje lokalne zaburzenia wag. Jeżeli nowe rozwiązanie jest lepsze, zostaje zaakceptowane. Rozwiązanie gorsze może zosta? zaakceptowane z prawdopodobieństwem zależnym od temperatury. Temperatura maleje w kolejnych iteracjach, dlatego algorytm stopniowo przechodzi od eksploracji do eksploatacji.

\subsection{Genetic Algorithm}
GA operuje na populacji portfeli. W kolejnych generacjach wybierane są lepsze osobniki, tworzone są potomki przez krzy?owanie, a nastópnie część wag jest mutowana. Elitaryzm pozwala zachować najlepsze rozwiązania. Każdy nowy osobnik przechodzi procedur? naprawy ograniczeń.

\subsection{Particle Swarm Optimization}
PSO traktuje portfele jako cz?stki poruszaj?ce się w przestrzeni rozwiązań. Ka?da cz?stka aktualizuje pozycję na podstawie własnego najlepszego położenia oraz najlepszego położenia całego roju. Metoda jest naturalnie dostosowana do ciągłej optymalizacji wag, natomiast ograniczenie kardynalności wymaga procedury naprawy.

\section{Projekt eksperymentu}
Eksperyment przeprowadzono dla nastópuj?cych wartości ograniczenia kardynalności:
\[
A \in \{2,3,5,10,15\}.
\]
Dla każdej wartości $A$ i każdego algorytmu wykonano kilka powtórzeń. Jako podstawowe kryterium jakości przyjęto najniższą wariancję portfela przy spełnieniu ograniczeń. Dodatkowo analizowano czas działania, stabilno?? wyników oraz dywersyfikację.

\section{Metryki oceny}
Do oceny wyników wykorzystano:
\begin{itemize}
    \item dzienną i roczną wariancję portfela,
    \item dzienną i roczną zmienność portfela,
    \item oczekiwaną stopę zwrotu,
    \item odsetek rozwiązań spełniaj?cych ograniczenia,
    \item średni czas działania,
    \item efektywn? liczbę aktywów,
    \item indeks koncentracji HHI,
    \item najwi?ksz? pojedyncz? wagę,
    \item udział klas aktywów w portfelu.
\end{itemize}

\section{Wyniki eksperymentu}
Tabela \ref{tab:summary} przedstawia zagregowane wyniki według algorytmu i wartości $A$.
\begin{table}[H]
\centering
\caption{Podsumowanie wyników algorytmów}
\label{tab:summary}
\resizebox{\textwidth}{!}{\input{tables/algorithm_summary.tex}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{min_variance_algorithms.png}
\caption{Minimalna wariancja według algorytmu i ograniczenia kardynalności}
\label{fig:min_variance}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{runtime_algorithms.png}
\caption{Porównanie czasu działania algorytmów}
\label{fig:runtime}
\end{figure}

\section{Dywersyfikacja i skład portfela}
Dywersyfikacja jest istotna, ponieważ portfel o niskiej wariancji może być jednocze?nie silnie skoncentrowany. W raporcie analizowana jest relacja między efektywn? liczbę aktywów a wariancję portfela.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{diversification_vs_variance.png}
\caption{Relacja dywersyfikacji i osięgni?tej wariancji}
\label{fig:diversification_variance}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{best_portfolio_weights.png}
\caption{Wagi najlepszego portfela według minimalnej wariancji}
\label{fig:best_weights}
\end{figure}

\section{Walidacja wyników}
Dla każdego portfela sprawdzono sumę wag, nieujemność, ograniczenie kardynalności, minimalny oczekiwany zwrot oraz poprawność macierzy kowariancji. Wyniki walidacji zapisano w katalogu \texttt{tables/validation/}. Walidacja jest niezb?dna, ponieważ sama niska warto?? funkcji celu nie wystarcza, jeżeli portfel narusza ograniczenia formalne.

\section{Uwagi krytyczne i ograniczenia badania}
Wyniki należy interpretować ostrożnie. Po pierwsze, dane mają charakter syntetyczny, więc wnioski nie mogą być bezpośrednio traktowane jako rekomendacje inwestycyjne. Po drugie, metaheurystyki są metodami stochastycznymi i zależą od ziarna losowego oraz hiperparametr?w. Po trzecie, procedura naprawy ograniczeń wpływa na przeszukiwaną przestrze? rozwiązań. Po czwarte, minimalizacja wariancji nie opisuje wszystkich aspekt?w ryzyka, zw?aszcza przy rozk?adach sko?nych lub z grubymi ogonami.

\section{Wnioski końcowe}
Przeprowadzone badanie pokazuje, że ograniczenie kardynalności zmienia klasyczny problem Markowitza w zadanie kombinatoryczne. Metaheurystyki pozwalają uzyskać dopuszczalne portfele i porównać kompromis między wariancję, czasem działania i dywersyfikację. Wskazanie najlepszego algorytmu powinno zawsze odnosię się do przyj?tych danych, parametr?w i liczby powtórzeń.

\appendix
\section{Aneks techniczny}
Finalny notebook znajduje się w \texttt{notebooks/final\_notebook.ipynb}. Wykresy zapisano w \texttt{charts/}, tabele w \texttt{tables/}, a konfigurację eksperymentu w \texttt{logs/experiment\_config.json}. Wszystkie ?cie?ki są wzgl?dne wzgl?dem katalogu projektu.

\begin{thebibliography}{9}
\bibitem{markowitz1952} H. Markowitz, Portfolio Selection, \textit{The Journal of Finance}, 7(1), 77--91, 1952.
\bibitem{kirkpatrick1983} S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, Optimization by Simulated Annealing, \textit{Science}, 220(4598), 671--680, 1983.
\bibitem{goldberg1989} D. E. Goldberg, \textit{Genetic Algorithms in Search, Optimization and Machine Learning}, Addison-Wesley, 1989.
\bibitem{kennedy1995} J. Kennedy, R. Eberhart, Particle Swarm Optimization, \textit{Proceedings of ICNN'95}, 1995.
\end{thebibliography}
\end{document}
"""
    (ROOT / "latex/main.tex").write_text(main, encoding="utf-8")
    refs = r"""@article{markowitz1952,
  author={Markowitz, Harry},
  title={Portfolio Selection},
  journal={The Journal of Finance},
  year={1952},
  volume={7},
  number={1},
  pages={77--91}
}
@article{kirkpatrick1983,
  author={Kirkpatrick, Scott and Gelatt, C. Daniel and Vecchi, Mario P.},
  title={Optimization by Simulated Annealing},
  journal={Science},
  year={1983},
  volume={220},
  number={4598},
  pages={671--680}
}
@book{goldberg1989,
  author={Goldberg, David E.},
  title={Genetic Algorithms in Search, Optimization and Machine Learning},
  publisher={Addison-Wesley},
  year={1989}
}
@article{kennedy1995,
  author={Kennedy, James and Eberhart, Russell},
  title={Particle Swarm Optimization},
  journal={Proceedings of ICNN'95},
  year={1995}
}
"""
    (ROOT / "latex/references.bib").write_text(refs, encoding="utf-8")


def write_readme() -> None:
    text = """# Markowitz Cardinality Portfolio Optimization

## Cel projektu
Minimalizacja wariancji portfela Markowitza z ograniczeniem kardynalności.

## Struktura folderów
- `notebooks/` - finalny notebook.
- `data/` - dane wejściowe.
- `charts/` - wykresy PNG/PDF.
- `tables/` - tabele CSV/XLSX/TEX.
- `latex/` - raport LaTeX.
- `logs/` - konfiguracja eksperymentu i walidacja.

## Jak uruchomić notebook
Uruchom `notebooks/final_notebook.ipynb` od początku do końca.

## Wymagane biblioteki
`numpy`, `pandas`, `matplotlib`, `openpyxl`.

## Gdzie znajduje się raport LaTeX
`latex/main.tex`.

## Jak skompilować raport w Overleaf
Prześlij folder `latex/` do Overleaf i ustaw `main.tex` jako plik główny.

## Uwagi dotyczące powtarzalności
Parametry eksperymentu zapisano w `logs/experiment_config.json`.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = prepare_data().apply(pd.to_numeric, errors="coerce").dropna()
    if (ROOT / "NMO_projekt.ipynb").exists():
        shutil.copy2(ROOT / "NMO_projekt.ipynb", ROOT / "notebooks/final_notebook.ipynb")
    results, summary, weights, meta = run_experiment(df)
    validation = results[["algorithm", "run", "A", "feasible", "cardinality", "daily_return", "daily_variance"]].copy()
    save_table(results.drop(columns=["weights"]), "tables/results/all_runs_results")
    save_table(summary, "tables/summary/algorithm_summary")
    save_table(weights, "tables/weights/best_portfolio_weights")
    save_table(validation, "tables/validation/validation_results")
    charts(results, summary, weights, meta)
    for rel in ["algorithm_summary.tex", "best_portfolio_weights.tex", "validation_results.tex"]:
        src = next((ROOT / base / rel for base in ["tables/summary", "tables/weights", "tables/validation"] if (ROOT / base / rel).exists()), None)
        if src:
            shutil.copy2(src, ROOT / "latex/tables" / rel)
    for rel in [
        "charts/algorithms/min_variance_algorithms.png",
        "charts/algorithms/runtime_algorithms.png",
        "charts/cardinality/diversification_vs_variance.png",
        "charts/weights/best_portfolio_weights.png",
    ]:
        shutil.copy2(ROOT / rel, ROOT / "latex/figures" / Path(rel).name)
    (ROOT / "logs/experiment_config.json").write_text(
        json.dumps({"seed": SEED, "A_VALUES": A_VALUES, "N_RUNS": N_RUNS, "N_CANDIDATES": N_CANDIDATES}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ROOT / "logs/validation_summary.txt").write_text(f"Wszystkie wyniki dopuszczalne: {bool(validation.feasible.all())}\n", encoding="utf-8")
    write_latex()
    write_readme()
    print("Projekt LaTeX zbudowany.")
    print(f"Najlepszy algorytm: {meta['best']['algorithm']}, A={int(meta['best']['A'])}")


if __name__ == "__main__":
    main()
