# data-science-notebooks

A collection of Python notebooks covering quantitative methods in economics, statistics, and data science — from experimental design to financial risk modeling.

Each folder corresponds to a topic and contains self-contained notebooks: theory is introduced before code, results are interpreted in context, and every notebook runs top-to-bottom without prior setup.

## Contents

| Folder | # | Notebook | Topics |
|--------|---|----------|--------|
| `04-econometria-basica` | 1 | `rct_difference_in_differences.ipynb` | RCT · DiD · Statistical Power |
| `04-econometria-basica` | 2 | `linear_regression_lasso_ridge.ipynb` | OLS · Lasso · Ridge · ENIGH 2022 |

*Folders follow a suggested learning sequence. More notebooks added regularly.*

## How to Use

Clone the repo and open any notebook locally:

```bash
git clone https://github.com/ivandeluna/data-science-notebooks.git
cd data-science-notebooks
```

Or open directly in Google Colab by navigating to any `.ipynb` file on GitHub and clicking **Open in Colab**.

## Requirements

- Python >= 3.9
- Core packages: `numpy` `pandas` `matplotlib` `seaborn` `scipy` `statsmodels` `scikit-learn`

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
```

## Data Sources

Some notebooks require external datasets. Each notebook includes a data section with download instructions.

| Notebook | Dataset | Source |
|----------|---------|--------|
| `linear_regression_lasso_ridge.ipynb` | ENIGH 2022 — `concentradohogar.csv` | [INEGI](https://www.inegi.org.mx/programas/enigh/nc/2022/#documentacion) |

## Author

**Iván de Luna** · [@ivandeluna](https://github.com/ivandeluna) · [ivandeluna.github.io](https://ivandeluna.github.io)