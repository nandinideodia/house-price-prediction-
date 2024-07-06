# House Price Prediction using Boston Dataset

## Overview

This project aims to predict house prices using the Boston Housing Dataset. The dataset contains information about various houses in Boston through different features. A regression machine learning model is trained to predict house prices based on these features.

## Dataset

The Boston Housing Dataset is a famous dataset from the 1970s. It contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It has 506 samples and 14 feature variables.

### Features

1. `CRIM`: per capita crime rate by town
2. `ZN`: proportion of residential land zoned for lots over 25,000 sq. ft.
3. `INDUS`: proportion of non-retail business acres per town
4. `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. `NOX`: nitric oxides concentration (parts per 10 million)
6. `RM`: average number of rooms per dwelling
7. `AGE`: proportion of owner-occupied units built prior to 1940
8. `DIS`: weighted distances to five Boston employment centers
9. `RAD`: index of accessibility to radial highways
10. `TAX`: full-value property tax rate per $10,000
11. `PTRATIO`: pupil-teacher ratio by town
12. `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town
13. `LSTAT`: % lower status of the population
14. `MEDV`: Median value of owner-occupied homes in $1000s (Target Variable)

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Load the dataset:

The dataset can be loaded using scikit-learn's datasets module:

```python
from sklearn.datasets import load_boston
boston = load_boston()
```

## Model Training

1. **Data Preprocessing**: Handle missing values, if any, and perform feature scaling.
2. **Train-Test Split**: Split the dataset into training and testing sets.
3. **Model Selection**: Use regression models like Linear Regression, Ridge, Lasso, etc.
4. **Model Training**: Train the model on the training data.
5. **Model Evaluation**: Evaluate the model using metrics like Mean Squared Error (MSE), R^2 score, etc.


## Results

The model's performance is evaluated using the Mean Squared Error (MSE) and R^2 score. These metrics give an understanding of how well the model is performing in predicting house prices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)

---

This README file provides a comprehensive guide to understanding, setting up, and running the house price prediction project using the Boston dataset.
