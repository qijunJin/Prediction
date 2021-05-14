<img src="images/logo_small.jpg" height=110>

# The Customer Datathon

[Bitphy](https://bitphy.com) and [Ubiqum Code Academy](https://ubiqum.com) are jointly hosting a datathon aimed at data analysis students. Our goal is to use data analysis and machine learning to understand and predict customer behaviour at retail shops.

The datathon will take place from 9am to 7pm on March 14th, 2019 at a joint location of Ubiqum's offices in Barcelona, Berlin and Amsterdam.

We have provided a dataset containing anonymized transactions carried out by individual identified customers from 25 *brick-and-mortar* retail shops specialised in groceries in the province of Barcelona. Such transactions are dated between January, 2016 and November, 2018.

# The problem: can we predict customer behaviour?

### Goal

Predict, for each of the 1190 customers in the [datasets/customers_val.txt](datasets/customers_val.txt) file, the date of the first purchase after November 1st, 2018, and the amount of such purchase.

### Validation metric

An individual customer submission will be considered correct if the predicted date matches the date of the real purchase and if the predicted billing's absolute error is at most 10€. Then we will look at accuracy: we will divide the number of correct predictions by 1190, thus obtaining a metric betweeen 0 and 1.


# Available data

All datasets provided for this datathon are contained in the `datasets/` folder. We briefly describe them below. For a deeper description of the available data , an initial exploration and a fitting of a dummy model are available [here](explo-dummy.ipynb).

Besides the provided datasets, teams are free to use whatever open data sets they might see fit, provided they satisfy [this](http://opendatahandbook.org/glossary/en/terms/open-data/) definition.

## Data for feature engineering and model training

- `customers.txt`: a list of the 8723 customer ids appearing in the dataset.

- `X.csv`: the basket-detail shopping history of customers appearing in `customers.txt` between the dates of January 1st, 2016 and October 31st, 2018. Shape: `(2281130, 9)`.

- `y.csv`: the date and billing of the first purchase effected by all customers appearing in `customers.txt` after November 1st, 2018. Shape: `(8723, 3)`.

## Validation data

- `customers_val.txt`: a list of the 1190 customer ids which have been selected to assess model accuracy in the competition.

- `X_val.csv`: the basket-detail shopping history of customers between the dates of January 1st, 2016 and October 31st, 2018. Shape: `(323227, 9)`.


# How to prepare a submission

Submissions are expected to be in the form of a dataframe with three columns labelled `customer`, `date` and `billing` and a total of 1190 rows.

An example of a valid submission may be found in the [datasets/sample_submission.csv](datasets/sample_submission.csv) file. Its first 5 rows look as follows:

customer | date | billing
---------|------|--------
sW2Tg | 2018-11-04 | 20.07
jL2H\_ | 2018-11-02 | 24.82
ows9Y | 2018-12-20 | 12.41
bNtuI | 2018-11-02 | 21.42
Bg9db | 2018-11-02 | 14.11


# Accuracy metric

A `score` method has been supplied in order to validate the outcome of a predictive model, and it is available as an importable module at [src/validations.py](src/validations.py). This is its signature:

```python
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
```

The example submission we just discussed comes from training a _dummy_ predictor. In order to get its score, we would run the following code:

```python
from src.validations import score

# Assuming y_real and y_pred contain the target
# values and the predictions, respectively.
accu = score(y_real, y_pred)
```

As a baseline for comparison, this dummy predictor attains a score of `accu = 0.046218`.



# How to submit a prediction

A live submission tracking system has been deployed by @gparrello. This [deployment](https://github.com/gparrello/grafana_api) uses a **REST API**. Gerardo has kindly provided code for quick result submission to this tracking system.

Each team will be provided with a configuration file containing a JWT (JSON Web Token) **token**, which will have to be passed to the submission function. An example of such a config file is available at [src/config.ini](src/config.ini). Download this file and save it in your computer.

- **R users**: use the `submit_predictions` function in the [src/client.R](src/client.R) file. An example of a code that uses `submit_predictions` to submit our sample submission may be found in [src/example.R](src/example.R).

- **python users**: use the `submit_predictions` method in the module [src/client.py](src/client.py). An example of a code that uses `submit_predictions` to submit our sample submission may be found in [src/example.py](src/example.py).



### License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
