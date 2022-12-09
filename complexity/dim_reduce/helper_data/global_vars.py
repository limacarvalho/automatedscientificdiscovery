
# how many cpus for parallel computing with ray
globalvar_n_cpus = 10

# how many gpus for parallel computing with ray
globalvar_n_gpus = 32

# timeout for ray in seconds, second step hyperparameter optimization
# the timeout for init and iter steps of hyperparameter optimization of one function
# which can be many dim reductions depending on the number of hyperparameters
globalvar_ray_timeout_step2 = 420

# timeout for ray in seconds, 3rd step full data
# timeout for one dim reduction pr function, ray is reinitialized in every step
globalvar_ray_timeout_step3 = 300

# identifier of loss function, other options: trustworthiness
globalvar_loss_function = 'rel_err'

# string to indicate first step of finding the intrinsic dimension
globalstring_step1 = '1_target_dim'

# string to indicate second step of finding the intrinsic dimension
globalstring_step2 = '2_functions'

# string to indicate third step of finding the intrinsic dimension
globalstring_step3 = '3_fulldata'


