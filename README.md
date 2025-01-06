# Linear Regression from Scratch

This project implements a simple linear regression model using gradient descent from scratch in Python. The implementation includes batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

## Dataset

- Predictor Variable: `./linearX.csv`
- Response Variable: `./linearY.csv`

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib

## Installation

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib
```

## Usage

1. Download the dataset and place `linearX.csv` and `linearY.csv` in the project directory.
2. Run the `linear_regression.py` script:

```bash
python linear_regression.py
```

## Description

### Preprocessing

- The dataset is loaded using pandas.
- The predictor variable is normalized.

### Gradient Descent Methods

1. **Batch Gradient Descent**:
   - Updates the parameters using the entire dataset in each iteration.

2. **Stochastic Gradient Descent**:
   - Updates the parameters using one randomly selected data point in each iteration.

3. **Mini-Batch Gradient Descent**:
   - Updates the parameters using a small random subset of the dataset in each iteration.

### Visualization

- The cost function vs. iterations is plotted for the first 50 iterations for all gradient descent methods.
- The linear regression fit is visualized for each gradient descent method.

## Results

The script prints the final parameters and cost for each gradient descent method and plots the following:
- Cost Function vs. Iterations (First 50 Iterations)
- Linear Regression Fit for Batch Gradient Descent
- Linear Regression Fit for Stochastic Gradient Descent
- Linear Regression Fit for Mini-Batch Gradient Descent

## License

This project is licensed under the MIT License.
