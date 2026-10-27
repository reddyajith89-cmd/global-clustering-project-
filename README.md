# Global Clustering Project

Groups the countries of the world by their development data using **PCA, KMeans and DBSCAN**,
and shows a **Streamlit app** where you can select countries and see their GDP for every year.

## Project files

| File | What it does |
|------|--------------|
| `train_model.py` | Cleans the data, runs PCA + KMeans + DBSCAN, saves results and models |
| `app.py` | Streamlit app (country selection + GDP graph + search box) |
| `requirements.txt` | Python libraries needed |
| `World_development_mesurement.xlsx` | Data file (put it in this folder) |
| `global_clustering_project_colab.ipynb` | Original Google Colab notebook |

## How to run (VS Code)

1. Open this folder in VS Code (`File` > `Open Folder`).
2. Open the terminal (`Ctrl + ~`) and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

3. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Put `World_development_mesurement.xlsx` in this folder
   (if it is missing, `train_model.py` tries to download it from GitHub).
5. Train the models (creates `country_results.csv`, `.pkl` files and the `plots/` folder):

   ```bash
   python train_model.py
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

   It opens at http://localhost:8501

## Push to GitHub

```bash
git init
git add .
git commit -m "Global clustering project"
git branch -M main
git remote add origin https://github.com/<your-username>/global-clustering-project.git
git push -u origin main
```

Tip: the `DATA_URL` inside `train_model.py` points to a GitHub raw link.
Change it to your own repository link after you upload the Excel file.
