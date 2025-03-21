import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def uncorrelated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Returns a subset of df columns with Pearson correlations below the threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold. Columns with correlation
                           higher than this value will be removed.

    Returns:
        pd.DataFrame: DataFrame with selected uncorrelated features.
    """

    corr = df.corr().abs()
    keep = []
    for i in range(len(corr.iloc[:, 0])):
        above = corr.iloc[:i, i]
        if len(keep) > 0:
            above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])

    return df[keep]


def diversity_filter(
    df: pd.DataFrame, diversity_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Returns a subset of df columns with diversity below the threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        diversity_threshold (float): Diversity threshold. Columns with diversity
                                        higher than this value will be removed.

    Returns:
        pd.DataFrame: DataFrame with selected diverse features.
    """

    diversities = []

    for columns in df.columns:
        uq = np.unique(df[columns])
        l = list(df[columns])

        dist = [l.count(uq[i]) / len(l) for i in range(len(uq))]

        qk = [1 / len(uq) for n in range(len(uq))]

        entropy_pk = entropy(dist, base=2)
        entropy_qk = entropy(qk, base=2)

        diversity = entropy_pk / entropy_qk
        diversities.append(np.round(diversity, 3))

    df_diversity = pd.DataFrame(
        data=diversities, columns=["Diversity"], index=df.columns
    ).sort_values(by="Diversity", ascending=True)

    to_drop = list(df_diversity[df_diversity["Diversity"] < diversity_threshold].index)

    df = df.drop(columns=to_drop)

    return df


def SFS(
    estimator,
    training_set_acceptor,
    test_set_acceptor,
    training_set_donor,
    test_set_donor,
    y,
    unseen_y,
    size,
    max_features,
    cv_iter,
):
    """A wrapper performing Sequential Forward Feature Selection. The rationale is to gradually increase the complexity of the models by adding one feature at a time.
    The wrapper will choose which feature to include in the process according to the highest gain in the cross-validation score.

    Arguments:

    estimator: a sklearn classifier
    training_set_acceptor: usually an empty pandas dataframe
    test_set_acceptor: usually an empty pandas dataframe
    training_set_donor: the training set whose features you want to gradually include (pandas dataframe)
    test_set_donor: the test set whose features you want to gradually include (pandas dataframe)
    y: the target variable in the training set (array, list)
    unseen_y: the target variable in the test set (array, list)
    size: the validation size during cross-validation (float, from 0 to 1)
    max_features: the maximum number of features the wrapper will try to obtain. Choose a small number to save computational cost. It cannot be larger than training_set_donor.shape[1]. (int)
    cv_iter: the number of iterations for cross-validation (int)"""

    scaler = StandardScaler()  # define scaler

    features = []  # where to store all feature subsets obtained at each iteration
    train_scores = (
        []
    )  # where to store all training scores obtained from the model using the accepted features
    test_scores = (
        []
    )  # w here to store all cross-validation scores obtained from the model using the accepted features

    test_unseen_scores = []  # where to store test score
    test_unseen_sens = []  # where to store sensitivity metric on the test set
    test_unseen_spec = []  # where to store specificity metric on the test set
    list_columns = list(training_set_donor.columns)

    for cycle in range(1, max_features):

        cv_scores_storage = []  # temporary list where to store cross-validation scores
        list_columns_shrink = [
            val for val in list_columns if val not in features
        ]  # list of features from which choose the best one at each cycle

        training_set_acceptor_copy = pd.concat(
            (training_set_acceptor, training_set_donor[features]), axis=1
        )  # growing training set receiving the new feature
        test_set_acceptor_copy = pd.concat(
            (test_set_acceptor, test_set_donor[features]), axis=1
        )  # growing test set receiving the new feature

        for (
            col
        ) in (
            list_columns_shrink
        ):  # for each new column, assess its potential inclusion:

            training_set_acceptor_temp = pd.concat(
                (training_set_acceptor_copy, training_set_donor[col]), axis=1
            )  # a temporary training set with a new feature to check
            test_set_acceptor_temp = pd.concat(
                (test_set_acceptor_copy, test_set_donor[col]), axis=1
            )  # a temporary test set with a new feature to check
            X = scaler.fit_transform(
                training_set_acceptor_temp
            )  # fit the scaler on the training set
            unseen_X = scaler.transform(
                test_set_acceptor_temp.values
            )  # scale the test set using mean and std of the training set
            estimator.fit(X, y)  # fit the estimator on the training set

            cv_scores_storage.append(
                np.mean(crossval_mcc(estimator, X, y, size, cv_iter))
            )  # compute and store cross-validation score

        # include as a feature the one providing the highest cross-validation score when used
        index = np.argmax(cv_scores_storage)
        feature_to_add = list_columns_shrink[index]
        features.append(feature_to_add)

        X = scaler.fit_transform(
            pd.concat((training_set_acceptor, training_set_donor[features]), axis=1)
        )  # fit the scaler on the training set with the new feature included
        unseen_X = scaler.transform(
            pd.concat((test_set_acceptor, test_set_donor[features]), axis=1).values
        )  # scale the test set with the new feature included
        estimator.fit(X, y)  # fit the estimator on the new training set

        train_scores.append(mcc(y, estimator.predict(X)))  # store training score
        test_scores.append(
            crossval_mcc(estimator, X, y, size, cv_iter)
        )  # store cv results for each feature subset
        estimator.fit(
            X, y
        )  # re-fit estimator otherwise it will be fitted on the last cv iteration

        confmat = confusion_matrix(
            y_true=unseen_y, y_pred=estimator.predict(unseen_X)
        )  # compute confusion matrix of the prediction on the test set
        tn = confmat[0][0]  # true negatives
        tp = confmat[1][1]  # true positives
        fn = confmat[1][0]  # false negatives
        fp = confmat[0][1]  # false positives

        test_unseen_scores.append(
            mcc(unseen_y, estimator.predict(unseen_X))
        )  # store test score
        test_unseen_sens.append(tp / (tp + fn))  # compute and store sensitivity
        test_unseen_spec.append(tn / (tn + fp))  # compute and store specificity

    return (
        train_scores,
        test_scores,
        test_unseen_scores,
        test_unseen_sens,
        test_unseen_spec,
        features,
    )  # return performance metrics and feature subsets


def crossval_mcc(estimator, X, y, size, n_iter):
    """Function computing the cross-validation MCC on the given dataset.

    Arguments:

    estimator: a sklearn classifier
    X: the training set (array)
    y: the target variable of the training set (array)
    size:  the validation size during cross-validation (float, from 0 to 1)
    n_iter: the number of iterations for cross-validation (int)"""

    scores = []
    cv_iter = StratifiedKFold(
        n_splits=n_iter, shuffle=True
    )  # define the cross-validation object

    cv_iter.split(X, y)  # split

    for train_index, test_index in cv_iter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        scores.append(mcc(y_test, estimator.predict(X_test)))

    return scores  # return the list of prediction performance across all iterations
