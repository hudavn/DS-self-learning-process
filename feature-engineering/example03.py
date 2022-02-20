# %%
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

# Lấy dữ liệu example từ package sklearn
X, y = make_classification()

print('X: \n', X[:5, :5])
print('y: \n', y)
print('X shape:', X.shape)
print('y shape:', y.shape)

# %%
VarianceThreshold(.5).fit_transform(X).shape

# %%
VarianceThreshold(0.9).fit_transform(X).shape

# %%
# So sánh 2 phương pháp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X_kbest = SelectKBest(f_classif, k = 5).fit_transform(X, y)
X_kvar = VarianceThreshold(0.9).fit_transform(X)
print('X shape after applying statistical selection: ',X_kbest.shape)
print('X shape after apply variance selection: ',X_kvar.shape)

# %%
# Hồi qui logistic
logit = LogisticRegression(solver='lbfgs', random_state=1)

# Cross validation cho:
# 1.dữ liệu gốc
acc_org = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Áp dụng phương sai
acc_var = cross_val_score(logit, X_kvar, y, scoring = 'accuracy', cv = 5).mean()
# 3. Áp dụng phương pháp thống kê
acc_stat = cross_val_score(logit, X_kbest, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy trên dữ liệu gốc:', acc_org)
print('Accuracy áp dụng phướng sai:', acc_var)
print('Accuracy dụng pp thống kê:', acc_stat)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Hồi qui theo RandomForest
rdFrt = RandomForestClassifier(n_estimators = 10, random_state = 1)

# Hồi qui theo LinearSVC
lnSVC = LinearSVC(C=0.01, penalty="l1", dual=False)

# Tạo một pipeline thực hiện lựa chọn biến từ RandomForest model và hồi qui theo logit
pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator = rdFrt), logit)

# Tạo một pipeline thực hiện lựa chọn biến từ Linear SVC model và hồi qui theo logit
pipe2 = make_pipeline(StandardScaler(), SelectFromModel(estimator = lnSVC), logit)

# Cross validate đối với 

# 1. Mô hình logit
acc_log = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Mô hình RandomForest
acc_rdf = cross_val_score(rdFrt, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe1
acc_pip1 = cross_val_score(pipe1, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe2
acc_pip2 = cross_val_score(pipe2, X, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy theo logit:', acc_log)
print('Accuracy theo random forest:', acc_rdf)
print('Accuracy theo pipeline 1:', acc_pip1)
print('Accuracy theo pipeline 2:', acc_pip2)

# %%
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(logit, scoring = 'accuracy', 
                                     verbose = 2, 
                                     k_features = 3,
                                     forward = False,
                                     n_jobs = -1)

selector.fit(X, y)