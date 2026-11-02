  # Model Comparison for Heart Disease Prediction

MCS 306: Introduction to AI, Bulacan State University.

Compares logistic regression, a decision tree, and K-nearest neighbors against a majority-class baseline on the Heart Failure Prediction Dataset (Fedesoriano, 2021).

## Structure
- `data/heart.csv`: dataset (918 patients, 11 predictors)
- `src/analysis.py`: full pipeline (tuning, nested CV, tests, figures)
- `results/`: metrics, nested CV scores, tree rules
- `figures/`: figures used in the paper
- `paper/main.tex`, `paper/main.pdf`: the paper

## Reproduce
```bash
pip install -r requirements.txt
python src/analysis.py
```

Seed 42 fixes the split, folds, and tree. Tested with scikit-learn 1.8.0.

## Data
Heart Failure Prediction Dataset by Fedesoriano (2021), from Kaggle:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Authors
Bon Federique B. Avendano, 
Leonicos Chael C. De Robles, 
Mark Angelo Del Rosario, 
Jerol Liuz V. Manlapig, 
Brylle James G. Robles

## License
Code and paper are released under the MIT License. The dataset belongs to its original source.
