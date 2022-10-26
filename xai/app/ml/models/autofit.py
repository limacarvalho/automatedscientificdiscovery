
from dask_ml.model_selection import train_test_split


class AutoFit:
    def __init__(self, df_X, df_y, pred_class, retrain=False) -> None:
        
        
        self.ensemble_n_trials = 200
        self.retrain = retrain        
        slug_ann.pred_class = pred_class
        self.df_X = df_X
        self.df_y = df_y
            
        
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=config.rand_state)

        X_scalar = StandardScaler().fit_transform(df_X.copy())

        X_base_scalar, X_test_scalar, y_base_scalar, y_test_scalar = train_test_split(X_scalar, df_y, test_size=0.33, random_state=config.rand_state)
        X_train_scalar, X_val_scalar, y_train_scalar, y_val_scalar = train_test_split(X_base_scalar, y_base_scalar, random_state=config.rand_state)

        self.slug_ann = SlugANN()
        self.slug_ann.pred_class = 'regression'
        self.slug_ann.X_train = X_train_scalar
        self.slug_ann.X_test = X_test_scalar
        self.slug_ann.X_val = X_val_scalar
        self.slug_ann.y_train = y_train_scalar
        self.slug_ann.y_test = y_test_scalar
        self.slug_ann.y_val = y_val_scalar
        self.slug_ann.n_trials = 20

        self.slug_xgb = SlugXGBoost()
        self.slug_xgb.pred_class = 'regression'
        self.slug_xgb.objective = 'count:poisson'
        self.slug_xgb.X_train = X_train
        self.slug_xgb.X_test = X_test
        self.slug_xgb.y_train = y_train
        self.slug_xgb.y_test = y_test
        self.slug_xgb.n_trials = 20

        base_models = [self.slug_ann, self.slug_xgb]
        
        self.base_unify = BaseUnify(timer=20, base_models=base_models, callback_timer_expired=None)
        

        
    def fetch_models(self):
        
        config.create_project_dirs(overwrite=self.retrain)
        
        self.base_unify.ensemble_n_trials = self.ensemble_n_trials
        
        self.base_unify.fetch_models(retrain=self.retrain)
        
        self.base_unify..fit_ensemble(retrain=self.retrain)
        
        

    def predict(df_X)    
        return self.base_unify.predict(df_X)