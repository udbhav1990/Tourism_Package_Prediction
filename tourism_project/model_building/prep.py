# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/udbhav90/tourism-package-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Binary features (can be treated as numeric too)
binary_features = [
    'Passport',
    'OwnCar'
]

# Define predictor matrix (X) using selected features
X = tourism_dataset[numeric_features + categorical_features + binary_features]

# Define target variable
y = tourism_dataset[target]

# Split dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

# Save splits to CSV
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="udbhav90/tourism-package-prediction",
        repo_type="dataset",
    )
