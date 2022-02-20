#%%
from main import dataset

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import beta
from scipy.stats import shapiro
import statsmodels.api as sm 
import numpy as np

# Tạo ra một chuỗi phân phối beta
data = beta(1, 10).rvs(1000).reshape(-1, 1)
print('data shape: %s'%str(data.shape))
# Sử dụng kiểm định shapiro để kiểm tra tính phân phối chuẩn.
shapiro(data)

# %%
shapiro(StandardScaler().fit_transform(data))

# %%
# biến đổi dữ liệu theo phân phối chuẩn:
price = np.float64(dataset.price.values)
print('Head 5 of original prices:', price[:5])
price_std = StandardScaler().fit_transform(price.reshape(-1, 1))
print('Head 5 of standard scaling prices:\n', price_std[:5])

# %%
price_mm = MinMaxScaler().fit_transform(price.reshape(-1, 1))
print('Head of min max scaling price:\n', price_mm[:5])

price_mm = (price - price.min())/(price.max() - price.min())
print('Head of min max scaling price:\n', price_mm[:5])