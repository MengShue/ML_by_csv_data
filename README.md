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

## Example: EEG Experiment for Left/Right Hand Movement

Imagine you are conducting an EEG experiment to classify imagined left and right hand movements:
	1.	Data Collection:
	•	Each subject participates in the experiment.
	•	EEG data is collected using multiple channels while subjects imagine moving their left or right hand.
	2.	Data Organization:
	•	Create one subfolder per subject (e.g., Subject_1, Subject_2).
	•	In each folder, create two .csv files:
	•	Left.csv for left-hand movement data.
	•	Right.csv for right-hand movement data.
	•	Each .csv file should have:
	•	Columns: EEG channels (e.g., Channel_1, Channel_2, …).
	•	Rows: EEG data samples.
	3.	Training and Prediction:
	•	Place the training folders in the root directory.
	•	Use a testing.csv file with new EEG samples to predict whether they correspond to left or right-hand movements.
	•	The program will dynamically process the data, train models, and output predictions.

```shell
data/
├── Subject_1/
│   ├── left.csv
│   └── right.csv
│
├── Subject_2/
│   ├── left.csv
│   └── right.csv
│
└── testing.csv
```
### EEG Data Structure

|   **Channel_1**   |   **Channel_2**   |   **Channel_3**   |   **...**   |
|:------------------:|:-----------------:|:-----------------:|:-----------:|
| EEG Data Sample 1  | EEG Data Sample 1 | EEG Data Sample 1 |    ...      |
| EEG Data Sample 2  | EEG Data Sample 2 | EEG Data Sample 2 |    ...      |
| EEG Data Sample 3  | EEG Data Sample 3 | EEG Data Sample 3 |    ...      |
|         ...         |         ...       |         ...       |    ...      |

## Example: Dielectric Material Resistance Experiment

In this experiment, we measure the electrical resistance of different dielectric materials under four different voltage settings. Each subfolder corresponds to a unique dielectric material, and each `.csv` file represents the data collected under a specific voltage setting.

#### Data Organization:
- **Folders**: Each folder represents a unique dielectric material resistance value.
- **CSV Files**: Each `.csv` file corresponds to one voltage setting (e.g., `Voltage_1.csv`, `Voltage_2.csv`).
- **Rows**: Each row represents a measurement value.
- **Columns**: Each column represents the experimental iteration.

#### Folder and File Structure:
```plaintext
data/
├── Material_1/
│   ├── Voltage_1.csv
│   ├── Voltage_2.csv
│   ├── Voltage_3.csv
│   └── Voltage_4.csv
├── Material_2/
│   ├── Voltage_1.csv
│   ├── Voltage_2.csv
│   ├── Voltage_3.csv
│   └── Voltage_4.csv
└── ...
```

### CSV File Structure

| **Experiment_1** | **Experiment_2** | **Experiment_3** |   **...**   |
|:----------------:|:----------------:|:----------------:|:-----------:|
|  Measurement_1   |  Measurement_1   |  Measurement_1   |    ...      |
|  Measurement_2   |  Measurement_2   |  Measurement_2   |    ...      |
|  Measurement_3   |  Measurement_3   |  Measurement_3   |    ...      |
|       ...        |       ...        |       ...        |    ...      |

## Example Output

- Training Data Summary:
- Dynamically detects N folders and M .csv files.
- Labels are automatically generated from file names.
- Best Model:
- Prints the name and accuracy of the best classifier.
- Testing Data Prediction:
- Outputs the predicted class of the testing data.

