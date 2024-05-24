import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

plt.style.use("tableau-colorblind10")


def dataset_summary(dataframe):
    """
    This function visually prints basic summary statistics of a given pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame  to examine.
    Returns
    -------
    None
        The function only prints the outputs to the console and doesn't return any value.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### NULL VALUES #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="viridis")
        plt.xticks(rotation=90)


def hyperparameter_optimization(X, y, models, cv=5, scoring="r2"):
    """
    Perform hyperparameter optimization for machine learning models using cross-validation.

    Parameters:
    -----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.

    models : list of tuples
        List containing tuples of model name, model object, and corresponding hyperparameter grid.

    cv : int, default=5
        Number of folds for cross-validation.

    scoring : str, default="r2"
        Scoring metric to optimize during hyperparameter tuning.

    Returns:
    --------
    best_models : dict
        Dictionary containing the best models found after hyperparameter optimization.
        Keys are model names and values are dictionaries with "Model" key for the optimized model object
        and "R2 Score" key for the mean R2 score obtained through cross-validation.

    Example:
    --------
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV

    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Define models with hyperparameter grids
    models = [
        ("Random Forest", RandomForestRegressor(), {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}),
        ("Lasso Regression", Lasso(), {'alpha': np.logspace(-4, 4, 10)})
    ]

    # Perform hyperparameter optimization
    best_models = hyperparameter_optimization(X, y, models, cv=5, scoring="r2")
    """
    print("Hyperparameter Optimization....")
    best_models = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model, params in models:
        print(name.center(len(name) + 2 * 10, "#"))
        cv_results = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        print(f"MCC (Before): {cv_results.mean()}")

        gs_best = GridSearchCV(model, params, cv=skf, n_jobs=-1, verbose=False).fit(X, y)
        final_model = model.set_params(**gs_best.best_params_)

        cv_results = cross_val_score(final_model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        print(f"MCC (After): {cv_results.mean()}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = {"Model": final_model, "MCC": cv_results.mean()}
    return best_models


def plot_roc_curve(model, X_test, y_test):
    """
    Function to plot the ROC curve for a given model and test data.

    Parameters:
    - model: The trained model that supports probability prediction.
    - X_test: Features from the test dataset.
    - y_test: True labels from the test dataset.
    """
    # Predict probabilities for the positive class
    y_scores = model.predict_proba(X_test)[:, 1]

    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


lr_params = {"penalty": ["l1", "l2"],
             "C": [0.001, 0.01, 0.1, 1, 10, 100],
             "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}

dt_params = {"max_depth": range(1, 20),
             "criterion": ["gini", "entropy"],
             "min_samples_leaf": [1, 2, 4]}

rf_params = {"max_depth": [None, 10, 20],
             "max_features": [5, 7, "auto"],
             "n_estimators": list(range(250, 500, 50)),
             "min_samples_leaf": [1, 2, 4]}

gb_params = {"n_estimators": list(range(250, 500, 50)),
             "max_depth": [5, 8, 12, None],
             "learning_rate": np.arange(0.05, 0.3, 0.05),
             "max_depth": [3, 5, 7]}

ada_params = {'n_estimators': list(range(250, 500, 50)),
              'learning_rate': np.arange(0.05, 0.5, 0.05),
              'algorithm': ['SAMME', 'SAMME.R'],
              'estimator': [None, DecisionTreeClassifier(), RandomForestClassifier()]}

svc_params = {'C': [0.1, 1, 10, 100],
              "kernel": ["linear", "rbf", "sigmoid"],
              "gamma": ["scale", "auto", 0.01, 0.1, 1, 10, 100]}

bagging_params = {"n_estimators": list(range(250, 500, 50)),
                  "max_samples": [0.4, 0.5, 0.6, 0.75, 1.0],
                  "max_features": [0.4, 0.5, 0.6, 0.75, 1.0],
                  "bootstrap": [True, False],
                  "bootstrap_features": [True, False]
                  }

knn_params = {"n_neighbors": list(range(2, 21)),
              "weights": ["uniform", "distance"],
              "metric": ["euclidean", "manhattan", "minkowski"]
              }

boost_params = {"learning_rate": np.arange(0.05, 0.5, 0.05),
                "max_depth": range(2, 10),
                "reg_alpha": [0, 0.1, 0.2],
                "reg_lambda": [0, 0.1, 0.2],
                "n_estimators": list(range(250, 500, 50)),
                "eval_metric": ["logloss"],
                "objective": ["binary:logistic"],
                "verbosity": [0]}

gbm_params = {"n_estimators": list(range(250, 500, 50)),
              "learning_rate": np.arange(0.05, 0.5, 0.05),
              "max_depth": range(2, 10),
              "verbose": [-1]
              }

classifiers = [("Logistic Regression", LogisticRegression(), lr_params),
               ("Decision Tree", DecisionTreeClassifier(), dt_params),
               ("Random Forest", RandomForestClassifier(), rf_params),
               ("Gradient Boosting", GradientBoostingClassifier(), gb_params),
               ("Ada Boosting", AdaBoostClassifier(), ada_params),
               ("Bagging", BaggingClassifier(), bagging_params),
               ("KNN", KNeighborsClassifier(), knn_params),
               ("SVM", SVC(), svc_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0), boost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), gbm_params)
               ]
