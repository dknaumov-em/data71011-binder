import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#set your own path to the babies data
babies = pd.read_table('babies.data.txt', sep=' ')
#view first 4 rows
babies.head(n=4)

#get number of observations
n = len(babies.index)

#convert into a 2D array (has rows and columns) for scikitlearn
mom_weight = babies['mom.weight'].values.reshape(-1, 1)
birth_weight = babies['birth.weight'].values
lm = LinearRegression()

#intercept only model
fit0 = lm.fit(np.ones((n, 1)), birth_weight)
r_sq = fit0.score(np.ones((n, 1)), birth_weight)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {fit0.intercept_}")
print(f"slope: {fit0.coef_}")
