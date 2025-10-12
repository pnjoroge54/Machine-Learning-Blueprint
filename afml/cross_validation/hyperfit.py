import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .cross_validation import PurgedKFold


class MyPipeline(Pipeline):
    """Allows for asample_weight in fit method"""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clf_hyper_fit(
    feat,
    lbl,
    t1,
    pipe_clf,
    param_grid,
    cv=3,
    bagging=[0, None, 1.0],
    rndSearchIter=0,
    n_jobs=-1,
    pctEmbargo=0,
    **fit_params,
):
    """
    hyperparameter search with purged k-fold cross-validation
    :param feat: features
    :param lbl: labels
    :param t1: t1 array
    :param pipe_clf: pipeline classifier
    :param param_grid: hyperparameter grid
    :param cv: number of cross-validation splits
    :param bagging: bagging parameters
    :param rndSearchIter: if 0 use GridSearchCV, else RandomizedSearchCV
    :param n_jobs: run in parallel if -1
    :param pctEmbargo: embargo on test set, default 0
    :param fit_params: fit parameters
    """
    if set(lbl.values) == {0, 1}:
        scoring = "f1"  # f1 for meta-labeling
    else:
        scoring = "neg_log_loss"  # symmetric towards all cases

    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    if rndSearchIter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            n_iter=rndSearchIter,
        )
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline

    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(
            estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat, lbl, sample_weight=fit_params[gs.estimator.steps[-1][0] + "__sample_weight"]
        )
        gs = Pipeline([("bag", gs)])

    return gs


def print_result(clf, n_splits):
    num_nodes = len(clf.cv_results_["rank_test_score"])
    total_time = clf.refit_time_ * num_nodes * n_splits

    best_scores, best_scores_idx, mean_score_time, mean_fit_time = [], 0, 0.0, 0.0

    for i in np.arange(n_splits):
        best_scores.append(clf.cv_results_["split" + str(i) + "_test_score"][clf.best_index_])
        idx = np.where(max(best_scores) == best_scores)[0][
            0
        ]  # always + 1 because index starts from 0 as default

    best_scores_idx = (clf.best_index_ + 1) + len(clf.cv_results_["mean_score_time"]) * idx
    print(
        f"Best params for estimator: {clf.best_estimator_} \nBest CV Score: {max(best_scores):.6f}\n"
    )
    print(
        f"A total of {best_scores_idx} was performed before optimal CV score found! (Under split{idx}_test_score / {idx + 1}th split)\n"
    )

    print(
        f"Estimated time taken for entire process: {total_time:.6f} seconds \nTotal number of candidates / nodes: {num_nodes}\n"
    )

    i = 0
    while i < (clf.best_index_ + 1):
        mean_score_time += clf.cv_results_["mean_score_time"][i]
        mean_fit_time += clf.cv_results_["mean_fit_time"][i]
        i += 1
    print("=" * 55)
    print(
        f"Estimated total mean time required for optimal solution: \n\nScore Time: {mean_score_time:.6f}s \nFit Time:  {mean_fit_time:.6f}s"
    )


def in_sample_sharpe_ratio(clf):
    sharpe_ratio = []
    for i in np.arange(len(clf.cv_results_["mean_test_score"])):
        m = np.sign(clf.cv_results_["mean_test_score"][i])
        sharpe_ratio.append(
            m * clf.cv_results_["mean_test_score"][i] / clf.cv_results_["std_test_score"][i]
        )
        # if clf.cv_results_['mean_test_score'][i] < 0:
        # sharpe_ratio.append(-1 * clf.cv_results_['mean_test_score'][i]/ clf.cv_results_['std_test_score'][i])
        # else:
        # sharpe_ratio.append(clf.cv_results_['mean_test_score'][i]/ clf.cv_results_['std_test_score'][i])
    print("IS Best Score Sharpe Ratio: {0:.6f}".format(sharpe_ratio[clf.best_index_]))
    print(
        f"Best IS Sharpe ratio: {max(sharpe_ratio):.6f} \nLowest IS Sharpe Ratio: {min(sharpe_ratio):.6f}"
        f"\nMean Sharpe Ratio: {np.mean(sharpe_ratio):.6f}"
    )


# in_sample_sharpe_ratio(clf)
