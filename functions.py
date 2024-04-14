"""
This file contains some helper functions to be used in the streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate


# custom scorer
def RMSE_log(y_true, y_pred):
    score = root_mean_squared_error(np.log(y_true), np.log(y_pred))
    return score


def run_model(X, y, pipeline_list, classifier, scorer, verbose=0, name=None):
    """
    Test the baseline case without any feature selection vs an sklearn classifier.  This function will take the 3 baseline cases (all features, nummeric PCA, and all features + numeric PCA)
    and return their CV and training score for each.

    Parameters:
        - X: the X matrix to be evaluated.
        - y: the y matrix to be evaluated.
        - pipeline list: A list of sklearn pipelines in the order of all 1) features  2) numeric PCA and 3) all features + numeric PCA
        - classifier: the Sklearn classifier object to be used to create the prediction.
        - scorer: the custom scorer used to create the CV and training scores.
        - name: this is the name (str) of the type of run. to be used for chart labels later.

    Returns:
        - training_score_list: The RMSE of the log prediction using the training data. will be a list of 3 values matching the baseline case list above.
        - CV_score_list: the RMSE of the log prediction using cross-validation.    will be a list of 3 values matching the baseline case list above.
        - train_score_stdev_list: standard deviation of the training scores.
        - C_score_stdev_list: standard deviation of the cross-validation scores.
    """

    # create the transform for the y value
    classifier_transform = TransformedTargetRegressor(
        regressor=classifier, func=np.log, inverse_func=np.exp
    )

    (
        training_score_list,
        CV_score_list,
        train_score_stdev_list,
        CV_score_stdev_list,
        estimators_list,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    print(f"Evaluating Model: {name}")
    for i, pipeline in enumerate(pipeline_list):
        clf = make_pipeline(pipeline, classifier_transform)

        # score the model
        cv_results = cross_validate(
            clf,
            X,
            y,
            cv=5,
            scoring=scorer,
            return_train_score=True,
            verbose=verbose,
            return_estimator=True,
        )

        CV_score_list.append(np.mean(cv_results["test_score"]))
        CV_score_stdev_list.append(np.std(cv_results["test_score"]))
        training_score_list.append(np.mean(cv_results["train_score"]))
        train_score_stdev_list.append(np.std(cv_results["train_score"]))
        estimators_list.append(cv_results["estimator"])

        # print(clf.fit(X, y)[:-1].get_feature_names_out()) #- can delete when done

        #print(f"-----Pipeline {i+1} complete------")

    return (
        training_score_list,
        CV_score_list,
        train_score_stdev_list,
        CV_score_stdev_list,
        estimators_list,
        name,
    )


def store_output(
    results_df, model_name, pipe_names, train_scores, CV_scores, train_std, CV_std
):
    """
    Take the results from a test iteration and append them to the results dataframe

    Parameters:
        - results_df:

    Returns:
        - The appended dataframe

    """

    # write outputs to dataframe
    index_names = [[model_name], pipe_names]
    index = pd.MultiIndex.from_product(index_names, names=["TestType", "FeaturesUsed"])

    data = {
        "train_scores": train_scores,
        "CV_scores": CV_scores,
        "train_std": train_std,
        "CV_std": CV_std,
    }

    result = pd.DataFrame(data=data, index=index)
    results = pd.concat([results_df, result])
    return results


# write a function to iterate over models and store their results
def iterate_over_models(classifier_tests, names, pipes, pipe_names, X_train, y_train, myscore):
    """
    This function will take in a list of regressors, iterate over and fit each model, and store the CV score and train score for each in a result data frame.

    Parameters:
        classifier_tests: A list of regressor objects to fit
        names: A list of strings describing the regression models
        pipes: A list of sklearn pipelines
        pipe_names: the list of strings that describe the number of features in the model (all original features, numeric PCA features only, all original + numeric PCA features)
        X_train: the X training matrix
        y_train: the y training matrix
        myscore: the scorer used for scoring the model.
        
    Returns:
        results: A results dataframe containing the training and CV scores.
    """
    results = pd.DataFrame()

    for classifier, name in zip(classifier_tests, names):
        train_scores, CV_scores, train_std, CV_std, estimators, name = run_model(
            X_train, y_train, pipes, classifier, myscore, name=name
        )

        # write outputs to dataframe
        results = store_output(
            results, name, pipe_names, train_scores, CV_scores, train_std, CV_std
        )

    return results


# write a function to plot the results
def plot_results(results, names, pipe_names):
    """
    This function will take the results dataframe and plot the training scores and CV scores of the fitted models as a bar chart.

    Parameters:
        results: the results dataframe containing the train scores and CV scores
        names: the list of strings that describe what classifiers have been fitted.
        pipe_names: the list of strings that describe the number of features in the model (all original features, numeric PCA features only, all original + numeric PCA features)

    Returns:
        Nothing, but a plot is displayed.
    """

    # data dictionaries used for plotting
    cv_scores_dict = {name: results["CV_scores"].loc[name].values for name in names}
    cv_std_dict = {name: results["CV_std"].loc[name].values for name in names}

    train_scores_dict = {
        name: results["train_scores"].loc[name].values for name in names
    }
    train_std_dict = {name: results["train_std"].loc[name].values for name in names}

    x = np.arange(len(pipe_names))  # label locations
    width = 0.1  # width of the bars
    multiplier = 0

    fig, axs = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    for test in names:
        offset = width * multiplier
        rects_cv = axs[1].bar(
            x + offset, cv_scores_dict[test], width, label=test, yerr=cv_std_dict[test]
        )
        rects_train = axs[0].bar(
            x + offset,
            train_scores_dict[test],
            width,
            label=test,
            yerr=train_std_dict[test],
        )
        labels_cv = axs[1].bar_label(rects_cv, padding=3, fmt="{:.3}")
        labels_train = axs[0].bar_label(rects_train, padding=3, fmt="{:.3}")
        [label.set_rotation(90) for label in labels_cv]
        [label.set_rotation(90) for label in labels_train]
        multiplier += 1

    axs[1].set_ylabel("RMSE")
    axs[0].set_ylabel("RMSE")
    axs[1].set_title("CV Scores")
    axs[0].set_title("Train Scores")
    axs[0].set_xticks(x + width, pipe_names)
    axs[1].set_xticks(x + width, pipe_names)
    axs[0].legend(loc="upper right")