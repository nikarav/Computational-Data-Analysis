import numpy as np


def accuracy_per_flight(actual_passengers, forecasted_passengers):
    deviation_p_f = (actual_passengers-forecasted_passengers)/actual_passengers
    return 1 - np.abs(deviation_p_f)
