import logging

from ray import tune
from relevance.utils import config
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from tune_sklearn import TuneSearchCV
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


class SlugRF:
    def __init__(
        self,
        name,
        pred_class,
        score_func=None,
        metric_func=None,
        max_depth=30,
        max_n_estimators=1000,
        n_trials=100,
        cv_splits=3,
        timeout=None,
    ) -> None:
        self.max_depth = max_depth
        self.max_n_estimators = max_n_estimators
        self.pred_class = pred_class
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
            model = RandomForestRegressor(random_state=config.rand_state, max_features=1.0)
            if self.metric_func is None:
                self.metric_func = r2_score

        else:
            model = RandomForestClassifier(random_state=config.rand_state, max_features=1.0)
            if self.metric_func is None:
                self.metric_func = f1_score

        return model

    def fit(self, X_train, X_test, y_train, y_test):
        logging.info(self.model_file_name + ": fit")

        param_dists = {
            "n_estimators": tune.randint(50, self.max_n_estimators),
            "max_depth": tune.randint(10, self.max_depth),
            "min_samples_split": tune.randint(2, 150),
            "min_samples_leaf": tune.randint(11, 60),
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
