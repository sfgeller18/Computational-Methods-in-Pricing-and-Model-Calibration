import numpy as np # for fast vector computations
import pandas as pd # for easy data analysis
import matplotlib.pyplot as plt # for plotting
from sklearn.linear_model import LinearRegression # for linear regression

df = pd.read_csv('swapLiborData.csv')
df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'],'D')
df.head()

def libor_rates_time_window(df, d1, d2):
    sub_df = pd.DataFrame()
    df2=df[(df['Date'] >= d1) & (df['Date'] <= d2)]
    sub_df=df2.iloc[:,1:6]    
    return sub_df

# we fix a time window
d1 = '2014-01-01'
d2 = '2016-05-24'
# we extract the data for the time window
sub_df = libor_rates_time_window(df, d1, d2)
sub_df.head()

#Linear Model Construction
y_name = sub_df['US0006M']
x_name = sub_df[['US0002M']]
print
print(x_name)
print(y_name)
b_0 = 0. # these are only initial values!
b_1 = 0.
R_2 = 0.

reg=LinearRegression()
reg.fit(x_name,y_name)
b_0=reg.intercept_
print(b_0)
b_1=reg.coef_
y_pred=reg.predict(x_name)
R_2=reg.score(x_name,y_name)

# Part 2
y_name = sub_df['US0012M']
x_name = sub_df[['US0002M', 'US0003M', 'US0006M']]
print(y_name)
print(x_name)
b_0 = 0. # these are only initial values!
b_1 = 0.
b_2 = 0.
b_3 = 0.
R_2 = 0.

reg=LinearRegression()
reg.fit(x_name,y_name)
b_0=reg.intercept_
b_1=reg.coef_
y_pred=reg.predict(x_name)
R_2=reg.score(x_name,y_name)

# we fix a time window
d1 = '2016-01-01'
d2 = '2017-12-31'

# we extract the data for the time window
sub_df = libor_rates_time_window(df, d1, d2)

def mean_sq_err(b_vector, x, y):
    # extract b's
    b_0 = b_vector[0]
    b_1 = b_vector[1]
    mse = 0.
    w=np
    w=np.power(np.add(y,-(b_0+b_1*x)),2)
    mse=np.average(w)  
    return mse

# define range for b_0 and b_1
lim = 200
space = 5
b_0_range = np.arange(-lim, lim + space, space)
len1 = len(b_0_range)
b_1_range = np.arange(-lim, lim + space, space)
len2 = len(b_1_range)
# here we store the values of f
f_grid = np.zeros((len2, len1))
# we create a grid
b_0_grid, b_1_grid = np.meshgrid(b_0_range, b_1_range)
# compute error surface
for i in range(len1):
    for j in range(len2):
        b_vec = np.array([b_0_grid[j, i], b_1_grid[j, i]])
        f_grid[j, i] = mean_sq_err(b_vec, x, y)

# plot the countours of the MSE
plt.figure(figsize=(8,5)) # set the figure size
plt.contour(b_0_grid, b_1_grid, f_grid,40)
plt.xlabel('b_0')
plt.ylabel('b_1')
plt.show()

def gradient_mse(b_vector, x, y):
    # extract b's
    b_0 = b_vector[0]
    b_1 = b_vector[1]
    grad = np.zeros(2)
    n=x.size
    X=np.sum(x)
    X_2=np.sum(np.power(x,2))
    X_3=np.sum(np.multiply(x,y))
    Y=np.sum(y)
    grad[0]=2*b_0+(2/n)*(b_1*X-Y)
    grad[1]=(2/n)*(b_1*X_2+b_0*X-X_3)
    return grad

def step_gd(b_vec_old, gamma, x, y):
    b_vec_new = np.zeros(2)
    g=gradient_mse(b_vec_old,x,y)
    b_vec_new[0]=b_vec_old[0]-gamma*g[0]
    b_vec_new[1]=b_vec_old[1]-gamma*g[1]
    return b_vec_new

def gradient_descent(b_vec0, gamma, m, x, y):    
    b_matrix = np.zeros([m + 1, 2])
    b_matrix[0] = b_vec0
    for k in range(m):
        b_matrix[k + 1] = step_gd(b_matrix[k], gamma, x, y)      
    return b_matrix

arr={0.005, 0.1, 0.55, 0.595}

for i in arr:
    gamma = 0.55
    b_vec_gd = gradient_descent(b_vec0, gamma, m, x, y)
    # this is the last point
    print('b_m = ' + str(b_vec_gd[-1]))
    # plot the countours of the MSE
    plt.figure(figsize=(8,5)) # set the figure size
    plt.contour(b_0_grid, b_1_grid, f_grid,40)
    plt.xlabel('b_0')
    plt.ylabel('b_1')

    # plot our GD iterations
    plt.plot(b_vec_gd[0,0], b_vec_gd[0,1], 'o')
    for k in range(m):
        plt.plot(b_vec_gd[k+1,0], b_vec_gd[k+1,1], 'o')
        plt.plot(b_vec_gd[k : k+2, 0], b_vec_gd[k : k+2, 1], '--r')
plt.show()

gamma=0.55
b_vec0 = [175, 25]
m = 300
b_vec_gd = gradient_descent(b_vec0, gamma, m, x, y)
print('b_0_star = ' + str(np.round(b_vec_gd[-1, 0], 4)))
print('b_1_star = ' + str(np.round(b_vec_gd[-1, 1], 4)))


