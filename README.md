# Support Instruments – Simulation Framework

## Dieses Projekt implementiert ein Framework zur Simulation von Handels- und Förderstrategien für erneuerbare Energien (Wind/Solar) in verschiedenen Förderregimen (z. B. FIT, Marktprämie, CfD).
## Datenbasis sind öffentlich verfügbare ENTSO-E Zeitreihen sowie zusätzliche Marktdaten (reBAP, ID1).

project-root/
│
├─ base/
│   └─ data/
│       reBAP_utc.csv         Rohdaten: Redispatch-Ausgleichspreise
│       id1_price_utc.xlsx    Rohdaten: ID1 Preise
│       data_final.csv        Cache (wird von build_cache.py erzeugt)
│
├─ results/                   Ausgabe-Verzeichnis (CSV-Ergebnisse)
│
├─ src/                       Zentrale Python-Module
│   __init__.py
│   config.py                 Globale Parameter & Spaltenkonventionen
│   data_import.py            Erstimport der ENTSO-E & Marktdaten aus DB/CSV/XLSX
│   data_store.py             Cache-Handling (CSV speichern/laden)
│   reduced_strategies.py     Kernlogik: Simulation der Förderregime
│   reduced_strategies_call.py Orchestrierung: mehrere Regime/Strategien
│
├─ scripts/                   Ausführbare Helfer/Analysen
│   build_cache.py            baut einmalig den CSV-Cache (data_final.csv)
│   reduced_strategies_index.py Sensitivitätsanalyse & Conformity-Index
│   smoke_test_core.py        Mini-Test: prüft Cache + Beispiel-Läufe
│
├─ .env                       DB-Credentials (nicht ins Repo pushen!)
├─ .gitignore                 ignoriert Cache-/Umgebungsdateien
└─ requirements.txt           Python-Abhängigkeiten


