"""
This file will take in the training data and save the cleansed/prepped X and y matrix as a pickle file.
The pickle files can then be used to laod the dataframe for the Streamlit app (app.py). 

This file will also create and store the pipeline objects to be used in the app as joblib files.
"""

import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from joblib import dump

# read data
df = pd.read_csv("train.csv")

# ensure all missing data coded the same
df.fillna(np.nan, inplace=True)

# drop the column MasVnrType as noted in the jupyter notebook
df.drop(columns=["MasVnrType"], inplace=True)

# create X and y matrices for modeling, drop Id column as it's not needed.
y = df["SalePrice"].copy()
X = df.drop(columns=["SalePrice", "Id"]).copy()

# save the X and y matrices to a pickle file.
y.to_pickle("./y.pkl")
X.to_pickle("./X.pkl")

# Create the pipeline objects to be used.
numeric_features = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

categorical_features = [
    "MSSubClass",  # was numeric but by description should be categorical
    "Alley",
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "Heating",
    "CentralAir",
    "Electrical",
    "Functional",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
    "Fence",
    "SaleType",
    "MiscFeature",
    "SaleCondition",
]

# ordinal features will be encoded, so initially need to define the categories for each feature
ordinal_features_dict = {
    "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex", np.nan],
    "PoolQC": ["Fa", "TA", "Gd", "Ex", np.nan],
    "BsmtQual": ["Po", "Fa", "TA", "Gd", "Ex", np.nan],
    "BsmtCond": ["Po", "Fa", "TA", "Gd", "Ex", np.nan],
    "BsmtExposure": ["No", "Mn", "Av", "Gd", np.nan],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ", np.nan],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ", np.nan],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex", np.nan],
    "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex", np.nan],
}

ordinal_features = list(ordinal_features_dict.keys())

# Case for only the original features
# for the ordinal features we will need a list of the categories so the order of the values can be defined.
categories = list(ordinal_features_dict.values())

# pipeline to for transforming original features only
features_only = make_pipeline(
    make_column_transformer(
        (
            SimpleImputer(strategy="constant", fill_value=0),
            numeric_features,
        ),  # fill 0 as described in above missing data section
        (
            TargetEncoder(categories="auto", target_type="continuous", random_state=41),
            categorical_features,
        ),
        (
            OrdinalEncoder(
                categories=categories, encoded_missing_value=-1
            ),  # encode na values as -1 as na values mean the absence of the basement, garage, etc...
            ordinal_features,
        ),
    ),
    StandardScaler(),
)

# Case for only PCA transformed numerical/ordinal columns

numerical_features_PCA = make_pipeline(
    make_column_transformer(
        (SimpleImputer(strategy="constant", fill_value=0), numeric_features),
        (
            OrdinalEncoder(categories=categories, encoded_missing_value=-1),
            ordinal_features,
        ),
        remainder="drop",  # drop the categorical features
    ),
    StandardScaler(),
    PCA(
        random_state=41,
    ),
)

# Case for the original featuers + all numeric/ordinal PCA transformed features
combined_features = make_union(features_only, numerical_features_PCA)

dump(features_only,"features_only.joblib")
dump(numerical_features_PCA, "numerical_features_PCA.joblib")
dump(combined_features, "combined_features.joblib")