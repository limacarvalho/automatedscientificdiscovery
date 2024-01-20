import logging

from ray import tune
from relevance.utils import config
from sklearn.metrics import f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tune_sklearn import TuneSearchCV
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


class BriskKNN:
    def __init__(
        self,
        name,
        pred_class,
        score_func=None,
        metric_func=None,
        n_neighbors=30,
        n_trials=100,
        cv_splits=3,
        timeout=None,
    ) -> None:
        self.pred_class = pred_class
        self.n_neighbors = n_neighbors
        self.n_trials = n_trials
        self.cv_splits = cv_splits  # number of folds
        self.random_state = config.rand_state

        self.model_file_name = name

        self.score_func = score_func
        self.metric_func = metric_func

        self.scores = []
        self.gs = None

        self.timeout = timeout

    def __get_model__(self):
        if self.pred_class == "regression":
            model = KNeighborsRegressor()
            if self.metric_func is None:
                self.metric_func = r2_score
        else:
            model = KNeighborsClassifier()
            if self.metric_func is None:
                self.metric_func = f1_score

        return model

    def fit(self, X_train, X_test, y_train, y_test):
        logging.info(self.model_file_name + ": fit")

        param_dists = {
            "n_neighbors": tune.randint(5, self.n_neighbors),
            "weights": tune.choice(["uniform", "distance"]),
            "algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
        }

        self.gs = TuneSearchCV(
            self.__get_model__(),
            param_dists,
            n_trials=self.n_trials,
            scoring=self.score_func,
            cv=self.cv_splits,
            search_optimization="hyperopt",
            time_budget_s=self.timeout,
        )

        self.gs.fit(X_train, y_train)

        pred_test = self.gs.predict(X_test)
        pred_train = self.gs.predict(X_train)

        err_train = self.metric_func(pred_train, y_train)
        err_test = self.metric_func(pred_test, y_test)

        self.scores = [err_train, err_test]

        logging.info(self.model_file_name + ": score: " + str(self.scores))

    def score(self, X, y, metric_func=None):
        if metric_func is None:
            metric_func = self.metric_func

        pred = self.gs.predict(X)

        return metric_func(pred, y)

    def predict(self, df_X):
        return self.gs.predict(df_X)
