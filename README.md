# Prediction
Prediction on purchase day

By far, I have tried to predict the date of the future purchase, but lack of knowledge, I couldn't get better solution for it.

Recency

![alt text](https://github.com/qijunJin/Prediction/blob/master/result/General/Recency.png)

Frequency

![alt text](https://github.com/qijunJin/Prediction/blob/master/result/General/Frequency.png)

Billing

![alt text](https://github.com/qijunJin/Prediction/blob/master/result/General/Billing.png)

Billing per month

![alt text](https://github.com/qijunJin/Prediction/blob/master/result/General/Billing_per_month.png)

My intention is to get 6 months before 01-nov to predict 3 months after 01-nov (as 75% of people do the next purchase within 64 days). Therefore, I get these data for each year seperately 2016, 2017, 2018. It includes recency, frequency, billing, clusters, difference between consecutive days from [6 months before 01-nov], mean and std. Finally for 2016 and 2017, I have also included the range of purchase (first day after 01-nov - last day before 01-nov).

For this anaysis, I have excluded chain, shop, seller and product.

Correlation matrix

![alt text](https://github.com/qijunJin/Prediction/blob/master/result/Result_Date/Corr.png)

Result

