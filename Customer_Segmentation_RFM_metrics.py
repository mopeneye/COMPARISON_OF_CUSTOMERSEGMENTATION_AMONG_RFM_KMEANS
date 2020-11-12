import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2010-2011 years data
df_2010_2011 = pd.read_excel('E:\PROJECTS\dsmlbc\CustomerSegmentation_kmeans\datasets\online_retail_II.xlsx', sheet_name = "Year 2010-2011")

df = df_2010_2011.copy() # do all things on df and backup first data

df["TotalPrice"] = df["Price"] * df["Quantity"] #TotalPrice calculation

df = df[~df["Invoice"].str.contains("C", na = False)] #data uncancelled

df.groupby(df["Invoice"]).agg({"TotalPrice":'sum'}).head() #totalPrices of Invoices

df.groupby(df["Country"]).agg({"TotalPrice":'sum'}).head()#country based sum of TotalPrices

df.groupby(df["Country"]).agg({"TotalPrice":'sum'}).sort_values("TotalPrice", ascending = False).head() # sorted

df.dropna(inplace = True) # drop of nan values

# Recency

df["InvoiceDate"].max() #nearest invoice date

today_date = dt.datetime(2011, 12, 9) # assumed today's date is the nearest invoice date in data

df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head() # customers' nearest visiting dates

df["Customer ID"] = df["Customer ID"].astype(int) #type cast for customer ID

# broadcasting feature has been used for subtracting nearest invoice date from today's date
today_date - df.groupby(df["Customer ID"]).agg({"InvoiceDate":'max'})


temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))

temp_df.rename(columns = {"InvoiceDate":"Recency"}, inplace = True) # Recency for all customers

recency_df = temp_df["Recency"].apply(lambda x: x.days) #day value calculation for all customers

# Frequency

# unique visit date counts of customers (frequencies) we can also use .count() instead of agg({"InvoiceDate":"nunique"})
freq_df = df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})

freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace=True)

# Monetary

monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"}) # monetary values per customers

monetary_df.rename(columns={"TotalPrice":"Monetary"}, inplace=True)

rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1) #rfm dataframe

rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4 , 3, 2, 1])
#sorted and calculated 5 quantile values for Recency  and labeled (descending, this is important!)

rfm["FrequencyScore"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
#sorted and calculated 5 quantile values for Frequency  and labeled (ascending, this is important!)

rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
#sorted and calculated 5 quantile values for Monetary  and labeled (ascending, this is important!)

(rfm['RecencyScore'].astype(str) +
 rfm['FrequencyScore'].astype(str) +
 rfm['MonetaryScore'].astype(str))

# rfm skorları kategorik değere dönüştürülüp df'e eklendi
rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str)) # RFM score calculation

# RFM segmentation score

# RFM Map via Regular Expressions
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

#we'll use only Recency and Frequency scores to calculate RFM Scores
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)# replate Segment values with Regex map

rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# K-MEANS

kmeans_df = rfm[['Recency', 'Frequency']]


# DATA PREPARATION

sc = MinMaxScaler((0, 1))
kmeans_transformed_df = sc.fit_transform(kmeans_df)

# K-MEANS

kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(kmeans_transformed_df)

# Optimum Küme Sayısının Belirlenmesi

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df1)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# K-means recommends 4 or 5 clusters for

kumeler = kmeans.labels_

kmeans_df["kmeans_cluster_no"] = kumeler

kmeans_df["kmeans_cluster_no"] = kmeans_df["kmeans_cluster_no"] + 1

rfm_last = rfm[['RFM_SCORE', 'Segment']]

rfm_last.index = rfm.index

df_both = pd.merge(kmeans_df, rfm_last, on=kmeans_df.index)

df_both.index = df_both.key_0
df_both.drop('key_0', axis=1, inplace=True)
df_both.index.name = 'Customer ID'
df_both.head()

# K-means clustering
df_both.groupby('kmeans_cluster_no').agg({'Recency': 'count'}).sort_values(by='Recency', ascending=False)

# RFM clustering
df_both.groupby('Segment').agg({'Recency': 'count'}) .sort_values(by='Recency', ascending=False)

# Comparison of K-means and RFM
for col in df_both['Segment'].unique():
    print(df_both[df_both['Segment'] == col].groupby(['Segment', 'kmeans_cluster_no']).agg({'Recency': 'count'}))


