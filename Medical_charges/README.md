# Medical Charges Prediction

## Project Description
  
The main objective is to **predict individual medical charges** (`charges`) billed by health insurance in the United States, based on personal and medical characteristics of the insured person.

## Dataset

- **File name**: `medical-charges.csv`
- **Number of observations**: 1,338
- **Number of variables**: 7 (6 features + 1 target)

### Features:

| Variable   | Type         | Description                          | Values |
|------------|--------------|--------------------------------------|--------|
| `age`      | Numeric      | Age of the insured                   | 18 - 64 |
| `sex`      | Categorical  | Gender                               | male, female |
| `bmi`      | Numeric      | Body Mass Index                      | 15.96 - 53.13 |
| `children` | Numeric      | Number of children/dependents        | 0 - 5 |
| `smoker`   | Categorical  | Smoking status                       | yes, no |
| `region`   | Categorical  | Residential area                     | northeast, northwest, southeast, southwest |
| `charges`  | Numeric      | **Target variable** - Medical charges| 1,121 - 63,770 |

## Methods Used

- **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate analysis, outlier detection
- **Data Preprocessing**: Handling duplicates, encoding categorical variables, feature scaling, train/test split
- **Modeling**: 
  - Linear Regression
  - Ridge Regression (L2 Regularization)
  - Lasso Regression (L1 Regularization)
  - ElasticNet Regression
- **Model Evaluation**: RMSE, MAE, R², Adjusted R², residual analysis
- **Feature Selection**: Backward Elimination and Lasso coefficients analysis
- **Comparison**: All features vs Selected features

## Best Model

**Linear Regression** achieved the best performance with **R² ≈ 0.8069**.

The most influential features are:
- `smoker`
- `age`
- `bmi`

## How to Run the Code

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
