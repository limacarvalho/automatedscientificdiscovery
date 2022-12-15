

class KnockoffSetting():
    SEED = 123
    ITR = 50
    FDR = 0.1
    # KSAMPLER = ['gaussian', 'metro']
    KSAMPLER = ['gaussian']
    # FSTATS  = ['lasso', 'ridge', 'deeppink', 'randomforest']
    FSTATS  = ['lasso', 'ridge', 'randomforest']

    class Config:
        case_sensitive = True


knockoffsettings = KnockoffSetting()

