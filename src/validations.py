"""Tools to validate the result outcomes from the datathon.
"""
import pandas as pd

def score(y, yhat):
    """Compute the score of predictions yhat by comparing to real data y.
    Both yhat and y have to be dataframes containing columns
        ['customer',
         'date',
         'billing']
    and must have exactly the same number of rows.

    A prediction for a given customer is computed as correct whenever the predicted date
    matches the real date and the predicted billing absolute error is lower or equal than
    10 €.

    :param y: (pd.DataFrame) real data.
    :param yhat: (pd.DataFrame) predicted data.
    :returns: score of the prediction.
    """

    # A few checks before going on with the computation.
    columns = [
        'customer',
        'date',
        'billing',
    ]
    for col in columns:
        if col not in yhat.columns:
            raise ValueError(
                f"Column {col} is missing from yhat."
            )
        if col not in y.columns:
            raise ValueError(
                f"Column {col} is missing from y."
            )

    if y.shape != yhat.shape:
        raise ValueError(
            f"Size mismatch: y.shape = {y.shape}, yhat.shape = {yhat.shape}"
        )

    big_y = pd.merge(y, yhat, how='outer', on='customer')
    big_y['delta_billing'] = big_y.billing_x - big_y.billing_y
    big_y['timedelta'] = big_y.apply(
        lambda row: (pd.to_datetime(row.date_x) - pd.to_datetime(row.date_y)).days,
        axis=1
    )

    mask = (abs(big_y['timedelta']) == 0) & (abs(big_y['billing_x'] - big_y['billing_y']) <= 10)
    return len(big_y[mask]) / len(y)
