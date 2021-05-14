import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans


%matplotlib inline
sns.set_style('darkgrid')


# X -> Data Structure & Parsing to DataFrame
dtype_X = {
    'transaction': np.int64,
    'customer': str,
    'chain': str,
    'shop': str,
    'seller': str,
    # 'timestamp': timestamp,
    'product': str,
    'quantity': np.float64,
    'billing': np.float64,
}
X = pd.read_csv('datasets/X.csv', dtype=dtype_X, parse_dates=['timestamp'])
X_original = X.copy()


# Y -> Data Structure & Parsing to DataFrame
dtype_y = {
    'customer': str,
    'billing': np.float64,
    # 'date': date,
}
y = pd.read_csv('datasets/y.csv', dtype=dtype_y, parse_dates=['date'])
y_original = y.copy()


# Y -> Transform date
y['date'] = y['date'].dt.date
y['date'] = y['date'].apply(lambda d: (d - y['date'].min()).days)
y = y.sort_values(by=['customer']).reset_index(drop = True)


# Customer ID -> Parsing to DataFrame
X_customer = pd.read_fwf("datasets/customers.txt", header=None, names=['customer']).sort_values(by=['customer']).reset_index(drop=True)
X['customer'].value_counts() # 'customer' is unique
X.customer.nunique()


# Consider this data as irrelevant
X["chain"].value_counts() # 9 -> Maybe relevant
X["shop"].value_counts() # 25 -> Irrelevant but considerable
X["seller"].value_counts() # 209 -> Irrelevant
X["product"].value_counts() # 11385 -> Irrelevant
X = X.drop(['seller', 'product', 'shop'], axis=1)


# Conversion timestamp -> date
X['date'] = X['timestamp'].dt.date
X.drop(['timestamp'], axis=1, inplace=True)
X.reset_index(drop=True, inplace=True)
X = X.sort_values(by=['customer', 'date'])


# Segmentation ordered
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


""" Only for global statistic
# Monthly billing total
X['year_month'] = X['date'].map(lambda date: date.replace(day=1))
X_revenue = X.groupby(['year_month'])['billing'].sum().reset_index()
plt.plot(X_revenue['year_month'], X_revenue['billing'])
plt.title("Billing per month")
plt.xticks(X_revenue['year_month'], X_revenue['year_month'], rotation='vertical')
plt.margins(0.05)
plt.show()


# Recency
X_recency = X.groupby('customer')['date'].max().reset_index()
X_recency['recency'] = (X_recency['date'].max() - X_recency['date']).dt.days
X_recency.describe() # mean 38, median 12
plt.hist(X_recency['recency'],[i for i in range(0,900,10)])
plt.title("Recency")
plt.show()

# Recency -> Segmentation
sse={}
X_recency_temp = X_recency[['recency']].copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_recency_temp)
    X_recency_temp["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_recency[['recency']])
X_recency['recencyCluster'] = kmeans.predict(X_recency[['recency']])

X_recency = order_cluster('recencyCluster', 'recency',X_recency,False)
X_recency.groupby('recencyCluster')['recency'].describe()
X_recency = X_recency.sort_values(by=['customer'])
# 3 -> best, 0 -> worst


# Frequency
X_frequency = X[['transaction', 'customer', 'date']].groupby(['transaction','customer']).first().reset_index()
X_frequency = X_frequency[['transaction', 'customer']].groupby(['customer']).count().reset_index().rename(columns = {'transaction': 'frequency'})
X_frequency.describe() # mean 54.3, median 41
plt.hist(X_frequency['frequency'], [i for i in range(0,190,10)])
plt.title("Frequency")
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_frequency[['frequency']])
X_frequency['frequencyCluster'] = kmeans.predict(X_frequency[['frequency']])

X_frequency = order_cluster('frequencyCluster', 'frequency',X_frequency,True)
X_frequency.groupby('frequencyCluster')['frequency'].describe()
X_frequency = X_frequency.sort_values(by=['customer'])
# 3 -> best, 0 -> worst


# Billing
X_billing = X[['transaction', 'customer', 'date', 'billing']].groupby(['transaction', 'customer', 'date']).agg('sum').reset_index()
X_billing = X_billing[['customer', 'billing']].groupby(['customer']).agg('sum').reset_index()
X_billing.describe() # mean 1261, median 849
plt.hist(X_billing['billing'], [i for i in range(0,22000,1000)])
plt.title("Billing")
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_billing[['billing']])
X_billing['billingCluster'] = kmeans.predict(X_billing[['billing']])

X_billing = order_cluster('billingCluster', 'billing',X_billing,True)
X_billing.groupby('billingCluster')['billing'].describe()
X_billing = X_billing.sort_values(by=['customer'])



# Overall
X_customer = pd.merge(X_customer, X_recency[['customer', 'recencyCluster']], on='customer')
X_customer = pd.merge(X_customer, X_frequency[['customer', 'frequencyCluster']], on='customer')
X_customer = pd.merge(X_customer, X_billing[['customer', 'billingCluster']], on='customer')
X_customer['overall'] = X_customer['recencyCluster'] + X_customer['frequencyCluster'] + X_customer['billingCluster']

X_customer = pd.merge(X_customer, X_recency[['customer', 'recency']], on='customer')
X_customer = pd.merge(X_customer, X_frequency[['customer', 'frequency']], on='customer')
X_customer = pd.merge(X_customer, X_billing[['customer', 'billing']], on='customer')
X_customer.groupby('overall')['recency','frequency','billing'].mean()
"""


# X6 to predict X3
X6_2016 = X[(X['date'] >= dt.date(2016,5,1)) & (X['date'] < dt.date(2016,11,1))].reset_index(drop=True)
X6_2017 = X[(X['date'] >= dt.date(2017,5,1)) & (X['date'] < dt.date(2017,11,1))].reset_index(drop=True)
X6_2018 = X[(X['date'] >= dt.date(2018,5,1)) & (X['date'] < dt.date(2018,11,1))].reset_index(drop=True)

def getLast(X6):
    X6_last = X6[['transaction', 'chain', 'customer', 'billing', 'date']].groupby(['transaction', 'chain', 'customer', 'date']).agg('sum').reset_index()
    X6_last = X6_last.sort_values(by=['customer', 'date'])
    X6_last = X6_last[['customer', 'date']].groupby('customer')['date'].max()
    X6_last = X6_last.reset_index().rename(columns = {'date': 'last_day'})
    return X6_last

X6_last_2016 = getLast(X6_2016)
X6_last_2017 = getLast(X6_2017)
X6_last_2018 = getLast(X6_2018)


# X3
X3_2016 = X[(X['date'] >= dt.date(2016,11,1)) & (X['date'] < dt.date(2017,2,1))].reset_index(drop=True)
X3_2017 = X[(X['date'] >= dt.date(2017,11,1)) & (X['date'] < dt.date(2018,2,1))].reset_index(drop=True)

def getFirst(X3):
    X3_first = X3[['transaction', 'chain', 'customer', 'billing', 'date']].groupby(['transaction', 'chain', 'customer', 'date']).agg('sum').reset_index()
    X3_first = X3_first.sort_values(by=['customer', 'date'])
    X3_first = X3_first[['customer', 'date']].groupby('customer')['date'].min()
    X3_first = X3_first.reset_index().rename(columns = {'date': 'first_day'})
    return X3_first

X3_first_2016 = getFirst(X3_2016)
X3_first_2017 = getFirst(X3_2017)


# Range before nov & after nov
def getNext(X6_last, X3_first):
    X_next = pd.merge(X6_last, X3_first, on='customer', how='left')
    X_next['next_day'] = (X_next['first_day'] - X_next['last_day']).dt.days
    return X_next

X_next_2016 = getNext(X6_last_2016, X3_first_2016)
X_next_2017 = getNext(X6_last_2017, X3_first_2017)

X_customer = pd.merge(X_customer, X_next_2016[['customer', 'next_day']], on='customer', how = 'left').rename(columns = {'next_day': 'next_day_2016'})
X_customer = pd.merge(X_customer, X_next_2017[['customer', 'next_day']], on='customer', how = 'left').rename(columns = {'next_day': 'next_day_2017'})

X_customer = X_customer.dropna()

X_customer_copy = X_customer['customer'].reset_index(drop = True)


# Recency
def recency(X6_last):
    X6_recency = X6_last.copy()
    X6_last_max = X6_last['last_day'].max()
    X6_recency['recency'] = X6_recency['last_day'].apply(lambda d: (X6_last_max - d).days)
    return X6_recency
    
X6_recency_2016 = recency(X6_last_2016)
X6_recency_2017 = recency(X6_last_2017)
X6_recency_2018 = recency(X6_last_2018)

X6_recency = pd.merge(X_customer_copy, X6_recency_2016[['customer', 'recency']], on='customer', how = 'left').rename(columns = {'recency': 'recency_2016'})
X6_recency = pd.merge(X6_recency, X6_recency_2017[['customer', 'recency']], on='customer', how = 'left').rename(columns = {'recency': 'recency_2017'})
X6_recency = pd.merge(X6_recency, X6_recency_2018[['customer', 'recency']], on='customer', how = 'left').rename(columns = {'recency': 'recency_2018'})

X6_recency = X6_recency.dropna()

def recency_kmeans(n, X6_recency, year):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X6_recency[['recency']])
    X6_recency['recencyCluster_' + str(year)] = kmeans.predict(X6_recency[['recency']])
    X6_recency = order_cluster('recencyCluster_' + str(year),'recency' ,X6_recency, True)
    X6_recency.groupby('recencyCluster_' + str(year))['recency'].describe()
    X6_recency = X6_recency.sort_values(by=['customer']).reset_index(drop = True)
    return X6_recency

X6_recency_kmean_2016 = recency_kmeans(10, X6_recency_2016, 2016)
X6_recency_kmean_2017 = recency_kmeans(10, X6_recency_2017, 2017)
X6_recency_kmean_2018 = recency_kmeans(10, X6_recency_2018, 2018)


# Frequency
def frequency(X6):
    X6_frequency = X6[['transaction', 'customer', 'date']].groupby(['transaction','customer']).first().reset_index()
    X6_frequency = X6_frequency[['transaction', 'customer']].groupby(['customer']).count().reset_index().rename(columns = {'transaction': 'frequency'})
    X6_frequency['frequency'] = -X6_frequency['frequency'] # Negative for corr
    return X6_frequency

X6_frequency_2016 = frequency(X6_2016)
X6_frequency_2017 = frequency(X6_2017)
X6_frequency_2018 = frequency(X6_2018)

X6_frequency = pd.merge(X_customer_copy, X6_frequency_2016[['customer', 'frequency']], on='customer', how = 'left').rename(columns = {'frequency': 'frequency_2016'})
X6_frequency = pd.merge(X6_frequency, X6_frequency_2017[['customer', 'frequency']], on='customer', how = 'left').rename(columns = {'frequency': 'frequency_2017'})
X6_frequency = pd.merge(X6_frequency, X6_frequency_2018[['customer', 'frequency']], on='customer', how = 'left').rename(columns = {'frequency': 'frequency_2018'})

X6_frequency = X6_frequency.dropna()

def frequency_kmeans(n, X6_frequency, year):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X6_frequency[['frequency']])
    X6_frequency['frequencyCluster_' + str(year)] = kmeans.predict(X6_frequency[['frequency']])
    
    X6_frequency = order_cluster('frequencyCluster_' + str(year), 'frequency',X6_frequency,True)
    X6_frequency.groupby('frequencyCluster_' + str(year))['frequency'].describe()
    X6_frequency = X6_frequency.sort_values(by=['customer'])
    return X6_frequency

X6_frequency_kmean_2016 = frequency_kmeans(10, X6_frequency_2016, 2016)
X6_frequency_kmean_2017 = frequency_kmeans(10, X6_frequency_2017, 2017)
X6_frequency_kmean_2018 = frequency_kmeans(10, X6_frequency_2018, 2018)


# Billing
def billing(X6):
    X6_billing = X6[['transaction', 'customer', 'date', 'billing']].groupby(['transaction', 'customer', 'date']).agg('sum').reset_index()
    X6_billing = X6_billing[['customer', 'billing']].groupby(['customer']).agg('sum').reset_index()
    X6_billing['billing'] = -X6_billing['billing']
    return X6_billing

X6_billing_2016 = billing(X6_2016)
X6_billing_2017 = billing(X6_2017)
X6_billing_2018 = billing(X6_2018)

X6_billing = pd.merge(X_customer_copy, X6_billing_2016[['customer', 'billing']], on='customer', how = 'left').rename(columns = {'billing': 'billing_2016'})
X6_billing = pd.merge(X6_billing, X6_billing_2017[['customer', 'billing']], on='customer', how = 'left').rename(columns = {'billing': 'billing_2017'})
X6_billing = pd.merge(X6_billing, X6_billing_2018[['customer', 'billing']], on='customer', how = 'left').rename(columns = {'billing': 'billing_2018'})

X6_billing = X6_billing.dropna()

def billing_kmeans(n, X6_billing, year):
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X6_billing[['billing']])
    X6_billing['billingCluster_' + str(year)] = kmeans.predict(X6_billing[['billing']])
    
    X6_billing = order_cluster('billingCluster_' + str(year), 'billing',X6_billing,True)
    X6_billing.groupby('billingCluster_' + str(year))['billing'].describe()
    X6_billing = X6_billing.sort_values(by=['customer'])
    return X6_billing

X6_billing_kmean_2016 = billing_kmeans(10, X6_billing_2016, 2016)
X6_billing_kmean_2017 = billing_kmeans(10, X6_billing_2017, 2017)
X6_billing_kmean_2018 = billing_kmeans(10, X6_billing_2018, 2018)



# Overall
X6_customer = X_customer_copy.copy()

def mergeClusters(X6_customer, X6_recency, X6_frequency, X6_billing, year):
    
    X6_customer = pd.merge(X6_customer, X6_recency[['customer', 'recencyCluster_' + str(year)]], on='customer')
    X6_customer = pd.merge(X6_customer, X6_frequency[['customer', 'frequencyCluster_' + str(year)]], on='customer')
    X6_customer = pd.merge(X6_customer, X6_billing[['customer', 'billingCluster_' + str(year)]], on='customer')
    X6_customer['overall'] = X6_customer['recencyCluster_' + str(year)] + X6_customer['frequencyCluster_' + str(year)] + X6_customer['billingCluster_' + str(year)]
    
    X6_customer = pd.merge(X6_customer, X6_recency[['customer', 'recency']], on='customer').rename(columns = {'recency': 'recency_' + str(year)})
    X6_customer = pd.merge(X6_customer, X6_frequency[['customer', 'frequency']], on='customer').rename(columns = {'frequency': 'frequency_' + str(year)})
    X6_customer = pd.merge(X6_customer, X6_billing[['customer', 'billing']], on='customer').rename(columns = {'billing': 'billing_' + str(year)})
    X6_customer.groupby('overall')[['recency_' + str(year),'frequency_' + str(year),'billing_' + str(year)]].mean()
    
    return X6_customer

X6_customer = mergeClusters(X6_customer, X6_recency_2016, X6_frequency_2016, X6_billing_2016, 2016)
X6_customer = mergeClusters(X6_customer, X6_recency_2017, X6_frequency_2017, X6_billing_2017, 2017)
X6_customer = mergeClusters(X6_customer, X6_recency_2018, X6_frequency_2018, X6_billing_2018, 2018)


# Day Difference
def dayDifference(X6):
    X6_diff = X6[['transaction', 'chain', 'customer', 'billing', 'date']].groupby(['transaction', 'chain', 'customer', 'date']).agg('sum').reset_index()
    X6_diff = X6_diff[['customer', 'date']]
    X6_diff = X6_diff.sort_values(by=['customer', 'date']).reset_index(drop=True)
    X6_diff = X6_diff.drop_duplicates(subset=['customer','date'],keep='first')
    
    X6_diff['1'] = X6_diff.groupby('customer')['date'].shift(1)
    X6_diff['2'] = X6_diff.groupby('customer')['date'].shift(2)
    X6_diff['3'] = X6_diff.groupby('customer')['date'].shift(3)
    X6_diff['4'] = X6_diff.groupby('customer')['date'].shift(4)
    X6_diff['5'] = X6_diff.groupby('customer')['date'].shift(5)
    X6_diff['6'] = X6_diff.groupby('customer')['date'].shift(6)
    
    X6_diff['d1'] = (X6_diff['date'] - X6_diff['1']).dt.days
    X6_diff['d2'] = (X6_diff['date'] - X6_diff['2']).dt.days
    X6_diff['d3'] = (X6_diff['date'] - X6_diff['3']).dt.days
    X6_diff['d4'] = (X6_diff['date'] - X6_diff['4']).dt.days
    X6_diff['d5'] = (X6_diff['date'] - X6_diff['5']).dt.days
    X6_diff['d6'] = (X6_diff['date'] - X6_diff['6']).dt.days
    
    X6_diff_features = X6_diff.groupby('customer').agg({'d1': ['mean','std']}).reset_index()
    X6_diff_features.columns = ['customer', 'mean','std']
    
    X6_diff_complete = X6_diff.drop_duplicates(subset=['customer'],keep='last')
    X6_diff_complete = X6_diff_complete.dropna()
    
    return X6_diff_complete, X6_diff_features

X6_diff_complete_2017, X6_diff_features_2017 = dayDifference(X6_2017)

X6_customer_all = X6_customer.copy()
X6_customer_all = pd.merge(X6_customer_all, X6_diff_complete_2017[['customer','d1', 'd2', 'd3', 'd4', 'd5', 'd6']], on='customer', how = 'left')
X6_customer_all = pd.merge(X6_customer_all, X6_diff_features_2017[['customer','mean', 'std']], on='customer', how = 'left')
X6_customer_all = pd.merge(X6_customer_all, X_next_2017[['customer','next_day']], on='customer', how = 'left')


X6_class = X6_customer_all.copy()

#X6_chain = X6[['customer', 'chain']].groupby(['customer']).first().reset_index()
#X6_chain = X6_chain.sort_values(by=['customer'])
#X6_class = pd.merge(X6_class, X6_chain[['customer','chain']], on='customer', how = 'left')


X6_class = pd.merge(X6_class, y[['customer','date']], on='customer', how = 'left')

X6_class= X6_class.drop(['customer'],axis=1)

X6_class = X6_class.dropna()

X6_class = pd.get_dummies(X6_class) # , drop_first = True


# Correlation
corr = X6_class[X6_class.columns].corr()
plt.figure(figsize = (30,20))
sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")

#X6_class= X6_class.drop(['overall'],axis=1)

# Model
X_val, y_val = X6_class.drop(['date'],axis=1), X6_class['date']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=0)
y_test= y_test.values


# Scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Linear
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test).round().astype(dtype = 'int64')

plt.scatter(y_test, y_pred)
ax = sns.regplot(x=y_test, y=y_pred,line_kws={"color": "red"} )
counter=0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        counter = counter+1
print(counter/len(y_test))


# Forest
from sklearn.ensemble import RandomForestRegressor
reg_forest = RandomForestRegressor(100)
reg_forest.fit(X_train, y_train)
y_pred_forest = reg_forest.predict(X_test).round().astype(dtype = 'int64')

plt.scatter(y_test, y_pred_forest)
ax = sns.regplot(x=y_test, y=y_pred,line_kws={"color": "blue"} )
counter=0
for i in range(len(y_test)):
    if y_test[i] == y_pred_forest[i]:
        counter = counter+1
print(counter/len(y_test))


# KNN
from sklearn import neighbors
n_neighbors = 7
reg = neighbors.KNeighborsRegressor(n_neighbors)
reg.fit(X_train, y_train)
y_pred_knn = reg.predict(X_test).round().astype(dtype = 'int64')

plt.scatter(y_test, y_pred_knn)
ax = sns.regplot(x=y_test, y=y_pred,line_kws={"color": "green"} )
counter=0
for i in range(len(y_test)):
    if y_test[i] == y_pred_knn[i]:
        counter = counter+1
print(counter/len(y_test))



