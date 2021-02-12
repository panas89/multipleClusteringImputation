# Multiple Clustering Imputation Process (MCIP)

This is a readme file that explains how to use the MCIP method.

## Create a Data directory and add the Athena dataset there.
- This is a directory that gets ignored by the git ignore file.
- All data will be stored there.

## Open Data Retrieval and Preprocessing.ipynb file and run the file.
- Missing values are assigned null values.
- Data split into training, validation and testing.
- Validation and testing sets missing value cases are filtered out.

## Open Test data Imputation method-ExternalTest and Validation combined  (40% test data).ipynb
- Load the MCIP module.
- Import training, validation and test sets.
- Generate random missing values to the validation and test sets.

## Open Test data Imputation method.ipnyb
- Load the MCIP module.
- MCIP imputation is based on the BallTree algorithm by [sklearn](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- Run a parallelized, over all cores of the computer, process that computes the imputation of missing values.
- Race variable defined based on most frequent race for all training, and testing datasets
- Imputed dataset saved.

## Open the GailRiskModel.ipnyb file (R based - needs R to be installed)
- Import the Imputed data and the original complete test data.
- Use the Gail model implemented by [Nova Smedley](https://github.com/novasmedley/Gail-Breast-Risk-Calculator).
- The computed 5 year risk is saved for both the imputed and complete test cases.

## Open the Results Analysis.ipnyb
- Histogram plots of risk for the complete and imputed test set.
- Calculation of HR,LR, U, HR-->LR, LR-->HR and LR/HR-->U cases.
- Calculation of the ratio of missing cases over all cases for each category of HR,LR, U, HR-->LR, LR-->HR and LR/HR-->U cases.


## To perform the four analyses of:

### No Missing Values, no variability

- Open Test data generation of random missing values.ipynb
- Set missingValues to **False**
- Open Test data Imputation method-ExternalTest and Validation combined  (40% test data).ipynb 
- Set partialLinear to **False**.

- Run the above notebooks in the order listed above.

### No Missing Values, with variability

- Open Test data generation of random missing values.ipynb
- Set missingValues to **False**
- Open Test data Imputation method-ExternalTest and Validation combined  (40% test data).ipynb 
- Set partialLinear to **True**.

- Run the above notebooks in the order listed above.

### Missing Values, no variability

- Open Test data generation of random missing values.ipynb
- Set missingValues to **True**
- Open Test data Imputation method-ExternalTest and Validation combined  (40% test data).ipynb 
- Set partialLinear to **False**.

- Run the above notebooks in the order listed above.

### Missing Values, with variability

- Open Test data generation of random missing values.ipynb
- Set missingValues to **True**
- Open Test data Imputation method-ExternalTest and Validation combined  (40% test data).ipynb 
- Set partialLinear to **True**.

- Run the above notebooks in the order listed above.
