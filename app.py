import streamlit as st
import pandas as pd


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

