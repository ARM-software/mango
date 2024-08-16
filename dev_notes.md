# 2024 Aug 16: Scaling parameters in GP space
Added an option `scale_params` to scale the parameters in GP space using min max scaler. Using 10 repeated `test_six_hump` test with param `y` scaled by a factor of 1000 we get the following results:

Base case (y param not scaled in inputs or GP space): 0 out of 10 tests failed
Y param not scaled, GP space scaled:                  1 out of 10 tests failed
Without scaling GP space:                             0 out of 10 tests failed 
With scaling GP space:                                2 out of 10 tests failed

Scaling is not clearly useful so not exposing it as an option. 