import streamlit as st
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression, SGDRegressor, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from functions import (
    RMSE_log,
    run_model,
    store_output,
    iterate_over_models,
    plot_results,
)


@st.cache_data
def get_pickle(filename):
    """Read a pickle file"""
    return pd.read_pickle(filename)


@st.cache_data
def get_joblib(filename):
    """Read a joblib file"""
    return load(filename)


@st.cache_data
def split_data(X, y):
    """Read in the X and y matrix and split the data using train_test_split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=38
    )
    return X_train, X_test, y_train, y_test


def main():
    # Read in necessary data files and split data
    X = get_pickle("./X.pkl")
    y = get_pickle("./y.pkl")
    features_only = get_joblib("./features_only.joblib")
    numerical_features_PCA = get_joblib("./numerical_features_PCA.joblib")
    combined_features = get_joblib("./combined_features.joblib")
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipes = [features_only, numerical_features_PCA, combined_features]
    pipe_names = ["features_only", "numeric_PCA", "combined"]
    myscore = make_scorer(RMSE_log)

    st.title("Feature Selection and Regularization (L1/L2)")
    st.subheader(
        "Their Influence on Predicted Housing Prices When Applying Regression Models",
        divider=True,
    )

    st.markdown(
        """
                Choose a model and parameters below to see how that model performs:
                * without feature selection or regularization (base case).
                * with the feature selection or regularization strategies you define.
                
                See how the parameters you selected compare with the parameters that I chose.       
                An example case is displayed below.  Feel free to modify to see the impact!
                """
    )
    model = st.radio(
        "Select the regression model to test",
        ["Simple Linear Regression", "Lasso", "Ridge", "SGDRegressor", "ElasticNet"],
    )

    st.text(f"You chose {model}.")

    # give some basic information about the selected model.
    match model:
        case "Simple Linear Regression":
            desc = """Simple Linear Regression is ordinarly least squares regression.
            While there is no built in feature seleciton with this model.  We can apply
            feature selection using sklearn's `SelectKBest`."""
            # more conditions here
        case _:
            desc = """Add more!"""

    st.markdown(desc)
    classifier_tests = [
        LinearRegression(),
        make_pipeline(SelectKBest(f_regression, k=3), LinearRegression()),
        make_pipeline(SelectKBest(f_regression, k=10), LinearRegression()),
        make_pipeline(SelectKBest(f_regression, k=20), LinearRegression()),
    ]
    names = [
        "Baseline Regression",
        "LR - 3 Best",
        "LR - 5 Best",
        "LR - 10 Best",
    ]

    with st.spinner("Fitting models..."):
        results = iterate_over_models(classifier_tests, names, pipes, pipe_names, X_train, y_train, myscore)
        fig = plot_results(results, names, pipe_names)
        st.pyplot(fig)
    st.success("Done!")


if __name__ == "__main__":
    main()
