# MOD 006570 — Edge-IIoTset intrusion detection and data poisoning

Coursework implementation for **Cybersecurity and AI Case Studies** using the **Edge-IIoTset** dataset (network intrusion / attack-type classification), plus **label poisoning** experiments and materials for the **lab logbook** and **Part C** design diagram.

Official dataset page: [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)  
Practical download mirror: [Kaggle — Edge-IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)

## Repository layout

| Path | Purpose |
|------|---------|
| `Coursework_MOD006570_EdgeIIoTset.ipynb` | Main **Jupyter** entry point (required submission artifact). |
| `run_coursework.py` | Same pipeline from the command line; also imported by the notebook. |
| `src/preprocess.py` | Loading + cleaning + dummy encoding aligned with authors’ DNN CSV guidance. |
| `src/evaluate.py` | Candidate models for **Part A** (tabular ML + optional XGBoost). |
| `src/poisoning.py` | **Part B** poisoning: random, systematic attack→benign, targeted borderline attack→benign. |
| `scripts/download_dataset.py` | Kaggle download helper for `DNN-EdgeIIoT-dataset.csv`. |
| `scripts/generate_notebook.py` | Regenerates the `.ipynb` if you edit its template. |
| `part_c_block_diagram.txt` | **Part C** — agents, secure channels, controls, Mermaid diagram draft. |
| `lab_logbook.txt` | **Part D** — ten session headings + what to evidence (figures, link to code). |
| `Report_File.txt` | Plain-language **how it works / how to run** (supporting document; not a substitute for the formal marked report). |
| `DATASET.txt` | Citation, DOIs, and dataset acquisition notes. |
| `SUBMISSION_CHECKLIST.txt` | Maps deliverables to the brief. |
| `data/` | Place `DNN-EdgeIIoT-dataset.csv` here (not committed; large). |
| `outputs/` | Generated tables and figures after a successful run. |

## Quick start



1. Install dependencies:

   ```powershell
   cd "c:\Users\Dheer Adarsh\Downloads\Cyber security\Cyber security"
   python -m pip install -r requirements.txt
   ```


   ```powershell
   python -m pip install numpy pandas scikit-learn matplotlib seaborn pycryptodome Pillow joblib
   ```

2. **Obtain the CSV** (choose one):

   - **Kaggle (recommended):** create `kaggle.json` under `%USERPROFILE%\.kaggle\`, then:

     ```powershell
     python scripts\download_dataset.py
     ```

   - **Manual:** download `DNN-EdgeIIoT-dataset.csv` and save as `data\DNN-EdgeIIoT-dataset.csv`.

3. **Run experiments:**

   ```powershell
   python run_coursework.py --data "data\DNN-EdgeIIoT-dataset.csv"
   ```

   Large-file tips:

   ```powershell
   python run_coursework.py --data "data\DNN-EdgeIIoT-dataset.csv" --nrows 250000 --fast
   ```

   - `--nrows` — cap rows for faster iteration.  
   - `--fast` — skips MLP and XGBoost.

4. **Jupyter:** open `Coursework_MOD006570_EdgeIIoTset.ipynb`, set `NROWS` / `FAST_MODELS` if needed, run all cells.

## Generate Week 1–10 lab outputs (screenshots / evidence)

This repo includes a small script that generates Week-style artifacts (text + charts) under `outputs/labs/`:

```powershell
python labs\run_all_labs.py
```

Outputs are written to `outputs/labs/` (e.g. `week1_split_shapes.txt`, `week4_mnist_fgsm.png`, `encrypted_ecb.png`, etc.).

## What gets generated (`outputs/`)

- **Part A:** `part_a_model_scores.csv`, `part_a_macro_f1_bar.png`, `part_a_binary_confusion.png`  
- **Part B:** `part_b_poisoning_metrics.csv` (includes `num_labels_changed`, `f1_drop_per_changed_label`), poisoning curves, `part_b_macro_f1_vs_num_changed.png`  
- **EDA:** `eda_attack_type_counts_raw.csv`, `eda_attack_type_counts_after_preprocess.csv`  
- **Run metadata:** `run_summary.json`

## Mapping to the case study

| Brief section | What this repo covers |
|---------------|------------------------|
| **Part A** | Multiple models, experimental comparison, test metrics; best model by macro-F1; binary Benign vs Attack confusion matrix for IDS interpretation. |
| **Part B** | Random flips, systematic attack→benign, targeted borderline attack→benign; sweeps 0–25% poison rate; plots + CSV including **absolute number of changed labels** (supports “minimal flips / maximal impact” analysis). |
| **Part C** | Text + Mermaid draft in `part_c_block_diagram.txt` — redraw as a proper figure in your report with literature references. |
| **Part D** | `lab_logbook.txt` template; paste `outputs/` figures and your **GitHub / OneDrive** code link. |
| **Formal report (Word/PDF)** | You still write the **marked report** yourself: Part A must be **model choice + experimental rationale + results only** (no introduction, literature survey, or conclusions, per brief). Use `Report_File.txt` as a technical companion, not as the final report. |



