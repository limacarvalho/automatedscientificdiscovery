
class DeepKnockoffSetting():
    SEED = 123
    ITR = 5
    FDR = 0.1

    EPOCHS = 100
    # Number of iterations over the full data per epoch
    EPOCH_LENGTH = 100
    # Data type, either "continuous" or "binary"
    # pars['family'] = "continuous" ### set later 
    # Dimensions of the data
    # pars['p'] = p
    # Size of the test set
    TEST_SIZE = 0
    # Batch size
    # pars['batch_size'] = int(0.5*n)
    # Learning rate
    LR = 0.01
    # When to decrease learning rate (unused when equal to number of epochs)
    LR_MILESTONE = EPOCHS
    # Width of the network (number of layers is fixed to 6)
    
    # Kernel widths for the MMD measure (uniform weights)
    ALPHAS = [1.,2.,4.,8.,16.,32.,64.,128.]

    MODELS = ['gaussian', 'gmm', 'mstudent', 'sparse', ]
    
    class Config:
        case_sensitive = True


deepknockoffsettings = DeepKnockoffSetting()

