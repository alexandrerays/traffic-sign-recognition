
import numpy as np
prediction_csv = np.empty([2, 2])
prediction_csv[0][0] = 1
prediction_csv[1][1] = 2
np.savetxt("test_" + str(1) + "_test.csv", prediction_csv.T, delimiter=",", fmt='%10.5f')