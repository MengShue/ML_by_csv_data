# Automated Machine Learning Classifier with Dynamic Labeling by CSV data format

This project provides a Python-based solution for dynamically loading data from multiple folders and using file names as labels for a machine learning classification task. The workflow is designed to automatically adapt to any changes in the folder and file structure, making it scalable and flexible.

## Features

- Dynamically detects subfolders and `.csv` files in a specified root directory.
- Uses file names as labels, automatically creating a mapping to numerical labels.
- Supports multiple machine learning classifiers (e.g., Logistic Regression, Random Forest, XGBoost, etc.).
- Automatically selects the best-performing classifier based on accuracy.
- Provides a pipeline to predict test data labels.

## Requirements

The project requires the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

You can install the required packages using:

```bash
pip install pandas numpy scikit-learn xgboost
```

## Folder and File Structure

1.	Training Data:
- place subfolders containing .csv files in the root directory (e.g., data/).
- Each subfolder represents a unique category.
- File names (excluding extensions) are used as labels.
2. Testing Data:
- Place a single testing.csv file in the root directory.

Example Structure:
```shell
data/
├── Folder_1/
│   ├── a.csv
│   ├── b.csv
│   ├── c.csv
│   └── d.csv
├── Folder_2/
│   ├── a.csv
│   ├── b.csv
│   ├── c.csv
│   └── d.csv
└── testing.csv
```
## How to Use

1. Prepare the Data:
- Create subfolders with .csv files in the root directory for training data.
- Ensure the testing.csv file is properly formatted with matching features.
2. Run the Script:
- Update the root_dir variable in the script to point to your data directory.
- Execute the script.
3.	Results:
- The script will output:
- An Excel file (combined_data_with_labels.xlsx) combining all training data with labels.
- The best-performing model and its accuracy.
- Predictions for the testing data.

## Output Files

- combined_data_with_labels.xlsx: Contains the combined training data with labels.
- Predictions: Outputs the predicted category for the test data in the terminal.

## Example Output

- Training Data Summary:
- Dynamically detects N folders and M .csv files.
- Labels are automatically generated from file names.
- Best Model:
- Prints the name and accuracy of the best classifier.
- Testing Data Prediction:
- Outputs the predicted class of the testing data.

