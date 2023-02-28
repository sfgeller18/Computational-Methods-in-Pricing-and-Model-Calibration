import numpy as np # for fast vector computations
import pandas as pd # for easy data analysis
import matplotlib.pyplot as plt # for plotting
from sklearn import linear_model # for linear regression

# load data using pandas
df = pd.read_csv('swapLiborData.csv')
df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'],'D')
df.head()

#Return LIBOR rates for a given date
def libor_rates_date(df, date):
    libor_rates = np.zeros(5)
    row=df[df['Date']==date].iloc[0]
    for i in range(5):
        libor_rates[i]=row[i+1]    
    return libor_rates

#Example Plot from LIBOR data
dates = ['2014-03-13', '2014-12-29', '2016-10-07', '2017-12-13', '2018-07-20']
plt.figure(figsize=(8,6)) # you can change the size to fit better your screen

for d in dates:
    plt.plot([1, 2, 3, 6 ,12], libor_rates_date(df, d)) # plot rates

# labels, title and legends
plt.xlabel('LIBOR term')
plt.ylabel('LIBOR rate')
plt.title('LIBOR Curve on various dates')
plt.legend(dates)
plt.show()

#Return LIBOR rate for given term at given dates
def libor_rate_date_term(df, date, term):
    libor = 0.
    s=np.array([1,2,3,6,12])
    l=libor_rates_date(df,date)
    for i in range(5):
        if term==s[i]:
            libor=l[i]
    return libor

#Return LIBOR rates for all terms between given dates
def libor_rates_time_window(df, d1, d2):
    sub_df = pd.DataFrame()
    df2=df[(df['Date'] >= d1) & (df['Date'] <= d2)]
    sub_df=df2.iloc[:,1:6]    
    return sub_df

#Plot Helper function
def scatter_plot_window(df, d1, d2):
    df_sub = libor_rates_time_window(df, d1, d2)
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(2,2,1)
    plt.title('Time window: ' + d1 + ' to ' + d2)
    plt.plot(df_sub.US0001M, df_sub.US0002M, '.')
    plt.xlabel('1M LIBOR rate')
    plt.ylabel('2M LIBOR rate')
    
    plt.subplot(2,2,2)
    plt.plot(df_sub.US0006M, df_sub.US0012M, '.')
    plt.xlabel('6M LIBOR rate')
    plt.ylabel('12M LIBOR rate')
    
    plt.subplot(2,2,3)
    plt.plot(df_sub.US0001M, df_sub.US0012M, '.')
    plt.xlabel('1M LIBOR rate')
    plt.ylabel('12M LIBOR rate')
    
    plt.subplot(2,2,4)
    plt.plot(df_sub.US0003M, df_sub.US0006M, '.')
    plt.xlabel('3M LIBOR rate')
    plt.ylabel('6M LIBOR rate')
    
    plt.show()

#Example Plot
scatter_plot_window(df, '2017-01-01', '2017-12-31')

#LIBOR correlation between two dates
def corr_window(df, d1, d2, term1, term2):
    corr = 0.0
    df1=libor_rates_time_window(df,d1,d2)
    print(df1.head(5))
    s=np.array([1,2,3,6,12])
    for i in range(5):
        if term1==s[i]:
            term1=i
        if term2==s[i]:
            term2=i
    df2=df1.iloc[:,term1]
    df3=df1.iloc[:,term2]
    print(df2.head(5))
    print(df3.head(5))
    corr=df2.corr(df3)    
    return corr

