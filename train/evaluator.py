import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "Pearson": np.corrcoef(y_true, y_pred)[0,1],
        "Spearman": stats.spearmanr(y_true, y_pred)[0]
    }
