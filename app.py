import streamlit as st
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import (
    Lasso,
    Ridge,
    LinearRegression,
    SGDRegressor,
    ElasticNet,
)
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

    # define the pipelines that will be used to fit the models below
    pipes = [features_only, numerical_features_PCA, combined_features]
    pipe_names = ["features_only", "numeric_PCA", "combined"]

    #define the scorer that will be used to score the models.
    myscore = make_scorer(RMSE_log)

    st.title("Feature Selection and Regularization (L1/L2)")
    st.subheader(
        "Their Influence on Predicted Housing Prices When Applying Regression Models",
        divider=True,
    )

    st.markdown(
        """
                Choose a model and parameters below to see how that model performs under the following conditions:
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
    st.divider()
    st.markdown("""Fill out the parameters below and then run your model.  
                The selections you make will define the feature selection or regularization parameters.
                See how your inputs compare to what I used!""")

    # create dynamic input
    match model:
        case "Simple Linear Regression":
            display_text = (
                "Choose how many features to select with SelectKBest using f_regression"
            )
            k1 = st.number_input(
                f"{display_text} (k1)",
                value=3,
                step=1,
                min_value=1,
            )
            k2 = st.number_input(
                f"{display_text} (k2)",
                value=5,
                step=1,
                min_value=1,
            )
            k3 = st.number_input(
                f"{display_text} (k3)",
                value=7,
                step=1,
                min_value=1,
            )
            regression_tests = [
                LinearRegression(),
                make_pipeline(SelectKBest(f_regression, k=k1), LinearRegression()),
                make_pipeline(SelectKBest(f_regression, k=k2), LinearRegression()),
                make_pipeline(SelectKBest(f_regression, k=k3), LinearRegression()),
            ]
            names = [
                "Baseline Regression",
                f"LR - {k1} Best (k1)",
                f"LR - {k2} Best (k2)",
                f"LR - {k3} Best (k3)",
            ]
            my_regression_tests = [
                LinearRegression(),
                make_pipeline(SelectKBest(f_regression, k=3), LinearRegression()),
                make_pipeline(SelectKBest(f_regression, k=10), LinearRegression()),
                make_pipeline(SelectKBest(f_regression, k=20), LinearRegression()),
            ]
            my_names = [
                "Baseline Regression",
                f"LR - {3} Best (k1)",
                f"LR - {10} Best (k2)",
                f"LR - {20} Best (k3)",
            ]
        case "Lasso":
            display_text = (
                "Choose an alpha value to apply to the Lasso regression."
            )
            a1 = st.number_input(
                f"{display_text} (alpha1)",
                value=1.0,
                step=0.01,
                min_value=0.0,
            )

            a2 = st.number_input(
                f"{display_text} (alpha2)",
                value=0.1,
                step=0.01,
                min_value=0.0,
            )
            
            a3 = st.number_input(
                f"{display_text} (alpha3)",
                value=0.01,
                step=0.01,
                min_value=0.0,
            )
            regression_tests = [
                LinearRegression(),
                Lasso(alpha=a1),
                Lasso(alpha=a2),
                Lasso(alpha=a3),
            ]
            names = [
                "Baseline Regression",
                f"Lasso alpha={a1}",
                f"Lasso alpha={a2}",
                f"Lasso alpha={a3}",
            ]
            my_regression_tests = [
                LinearRegression(),
                Lasso(alpha=1.0),
                Lasso(alpha=0.1),
                Lasso(alpha=0.01),
            ]
            my_names = [
                "Baseline Regression",
                "Lasso alpha=1.0",
                "Lasso alpha=0.1",
                "Lasso alpha=0.01",
            ]
        case "SGDRegressor":
            display_text_1 = ("Choose an alpha value")
            display_text_2 = ("Choose a penalty type (L1 or L2)")
            col1, col2 = st.columns(2)

            with col1:
                a1 = st.number_input(
                    f"{display_text_1} (a1)",
                    value=1.0,
                    step=0.01,
                    min_value=0.0,
                )

                a2 = st.number_input(
                    f"{display_text_1} (a2)",
                    value=0.1,
                    step=0.01,
                    min_value=0.0,
                )
                
                a3 = st.number_input(
                    f"{display_text_1} (a3)",
                    value=0.01,
                    step=0.01,
                    min_value=0.0,
                )

            with col2:
                penalty_1 = st.selectbox(
                    display_text_2,
                    ("L1", "L2"),
                    key="penalty1",
                )

                penalty_2 = st.selectbox(
                    display_text_2,
                    ("L1", "L2"),
                    key="penalty2",
                )

                penalty_3 = st.selectbox(
                    display_text_2,
                    ("L1", "L2"),
                    key="penalty3",
                )

                regression_tests = [
                    SGDRegressor(penalty=None, random_state=41),
                    SGDRegressor(penalty=penalty_1.lower(), alpha=a1, random_state=41),
                    SGDRegressor(penalty=penalty_2.lower(), alpha=a2, random_state=41),
                    SGDRegressor(penalty=penalty_3.lower(), alpha=a3, random_state=41),
                ]
                names = [
                    "Baseline SGDRegressor",
                    f"SGD alpha={a1} {penalty_1}",
                    f"SGD alpha={a2} {penalty_2}",
                    f"SGD alpha={a3} {penalty_3}",
                ]
                my_regression_tests = [
                    SGDRegressor(penalty=None, random_state=41),
                    SGDRegressor(penalty='l1', alpha=1.0, random_state=41),
                    SGDRegressor(penalty='l2', alpha=1.0, random_state=41),
                    SGDRegressor(penalty='l2', alpha=0.1, random_state=41),
                ]
                my_names = [
                    "Baseline SGDRegressor",
                    "SGD alpha=1.0 L1",
                    "SGD alpha=1.0 L2",
                    "SGD alpha=0.1 L2",
                ]



    st.markdown("Click button below to run model with above parameters.")
    if st.button("Run model"):
        with st.spinner("Fitting your models..."):
            results = iterate_over_models(
                regression_tests, names, pipes, pipe_names, X_train, y_train, myscore
            )
            st.markdown(f"Your model with the above feature selection/regularization parameters")
            fig = plot_results(results, names, pipe_names)
            st.pyplot(fig)
        
        # below are the models I ran
        st.markdown(f"Here is the chart for the {model} model using the feature selection/normalization parameters that I chose.")
        with st.spinner("Fitting my models..."):
            my_results = iterate_over_models(
                my_regression_tests, my_names, pipes, pipe_names, X_train, y_train, myscore
            )
            fig2 = plot_results(my_results, my_names, pipe_names)
            st.pyplot(fig2)
        st.success("Done!")


        st.markdown("""As can be seen by the above, in almost all cases the model using the combined pipeline (i.e. the case with all original features + the PCA transformed numerical features) along
                    with some form of feature selection or normalization has a better overall CV RMSE score along with a smaller standard deviation.""")

if __name__ == "__main__":
    main()
