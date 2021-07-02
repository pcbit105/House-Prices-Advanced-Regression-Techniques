import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math
import seaborn as sns


# SECTION 1: DATA PREPROCESSING AND FEATURE ENGINEERING
# Import csv file, preprocessing through dummy variables, create X and y datasets
dataset = pd.read_csv('train.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
# Dropping the ID column
dataset = dataset.iloc[:,1:]

# Find the data types of all dataframe columns
print(dataset.info())

# Convert MSSubClass to object data type (because it is categorical in nature despite having numbers)
dataset = dataset.astype({'MSSubClass':'object'})

# Examining correlation for continuous numerical variables, aiming to find variables with low correlation with SalePrice
print(dataset.corr(method='pearson'))

# Create list of columns that are numerical
numerical_columns = list(dataset.select_dtypes(include=['int64','float64']).columns)
numerical_columns

# For numerical columns, fill NA values with 0
for column in numerical_columns:
    dataset[column].fillna(0, inplace=True)

# List the correlation coefficients and p-values of each numerical column with SalePrice
from scipy import stats
pearson_coef_list = []
p_value_list = []
for column in numerical_columns:
    pearson_coef, p_value = stats.pearsonr(dataset[column], dataset['SalePrice'])
    pearson_coef_list.append(pearson_coef)
    p_value_list.append(p_value)
    print(column, ' - CC: ', round(pearson_coef,5), ', P-Value: ', round(p_value,8))
cc_p = pd.DataFrame(list(zip(pearson_coef_list,p_value_list)), index=numerical_columns, 
                    columns=['Correlation Coefficient','P-Value'])
pd.set_option('display.max_rows', len(cc_p))
cc_p

# Create list of numerical columns with low correlation coefficients and high p-value - we will drop these
numerical_columns_to_drop = []
for column in numerical_columns:
    if (cc_p.loc[column][0] < 0.1) and (cc_p.loc[column][0] > -0.1) and (cc_p.loc[column][1] > 0.05):
        numerical_columns_to_drop.append(column)

# Creating list of column names for 'object' data types
categorical_columns = list(dataset.select_dtypes(include=['object']).columns)
categorical_columns

# Create boxplot between categorical features and SalePrice
for column in categorical_columns:
    plt.figure()
    sns.boxplot(x=column, y="SalePrice", data=dataset)

# By observing boxplots, create list of categorical columns with high overlap for corresponding SalePrice value
categorical_columns_to_drop = []
categorical_columns_to_drop.extend(['LotShape', 'Utilities', 'LotConfig', 'LandSlope',
                                 'BsmtFinType1', 'BsmtFinType2'])

# Dropping the chosen numerical and categorical columns
dataset_cleaned = dataset.drop(numerical_columns_to_drop + categorical_columns_to_drop, axis=1)

# Divide into X and y sub data sets
X = dataset_cleaned.iloc[:,:-1]
y = dataset_cleaned.iloc[:,-1]

# Creating dummy variables for categorical columns
X = pd.get_dummies(X, columns=[column for column in categorical_columns if column not in categorical_columns_to_drop])

# Feature Scaling for numerical columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
X_scaled = X.copy()
numerical_column_count = len(numerical_columns) - len(numerical_columns_to_drop)
X_scaled.iloc[:,:numerical_column_count-1] = sc.fit_transform(X_scaled.iloc[:,:numerical_column_count-1])

# Split X and y datasets into testing and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, X_scaled, y, test_size=0.3, random_state=0)


# SECTION 2: MACHINE LEARNING ALGORITHM DEVELOPMENT AND PERFORMANCE ASSESSMENT
# Creating stratified K-folds for cross validation during hyperparameter optimization
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# Function to calculate RMSLE
from sklearn.model_selection import cross_validate
def rmsle(model):
    msle = cross_validate(model, X_test_scaled, y_test, cv = kfold, scoring='neg_mean_squared_log_error')['test_score'].mean()
    rmsle = (np.sqrt(msle * -1))
    return(rmsle)


# CatBoost with Optuna hyperparameter optimization framework
from catboost import CatBoostRegressor
from optuna import create_study
def cb_objective(trial):
    iterations = int(trial.suggest_loguniform('iterations', 300, 400))
    max_depth = int(trial.suggest_loguniform('max_depth', 3, 10))
    eval_metric = trial.suggest_categorical('eval_metric', ['MSLE','RMSE'])
    model = CatBoostRegressor(iterations = iterations, 
                              max_depth = max_depth,
                              eval_metric = eval_metric
                              )
    scores = cross_validate(model, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(cb_objective, n_trials=20)
best_trial_cb =  study.best_trial
print('CatBoost Optuna Best Trial')
print('RMSLE: {}'.format(np.sqrt(best_trial_cb.value * -1)))
print('Best params: {}'.format(best_trial_cb.params))

# Creating the optimized CatBoostRegressor model
iterations = int(best_trial_cb.params['iterations'])
max_depth = int(best_trial_cb.params['max_depth'])
eval_metric = best_trial_cb.params['eval_metric']
cb_optimal = CatBoostRegressor(learning_rate=0.1, 
                     iterations=iterations,
                     max_depth=max_depth,
                     eval_metric=eval_metric
                     )
print('CatBoost Optimal Model Performance')
print('RMSLE: {}'.format(rmsle(cb_optimal)))


# XGBoost with Optuna hyperparameter optimization framework
from xgboost import XGBRegressor
def xgb_objective(trial):
    n_estimators = int(trial.suggest_loguniform('n_estimators', 100, 300))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 10))
    booster = trial.suggest_categorical('booster', ['gbtree'])
    tree_method = trial.suggest_categorical('tree_method', ['hist'])
    model = XGBRegressor(n_estimators=n_estimators, 
                          max_depth=max_depth,
                          booster = booster,
                          tree_method = tree_method
                          )
    scores = cross_validate(model, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=20)
best_trial_xgb =  study.best_trial
print('XGBoost Optuna Best Trial')
print('RMSLE: {}'.format(np.sqrt(best_trial_xgb.value * -1)))
print('Best params: {}'.format(best_trial_xgb.params))

# Creating the optimized XGBRegressor model
n_estimators = int(best_trial_xgb.params['n_estimators'])
max_depth = int(best_trial_xgb.params['max_depth'])
booster = best_trial_xgb.params['booster']
tree_method = best_trial_xgb.params['tree_method']
xgb_optimal = XGBRegressor(n_estimators=n_estimators,
                     max_depth=max_depth,
                     booster=booster,
                     tree_method=tree_method
                     )
print('XGBoost Optimal Model Performance')
print('RMSLE: {}'.format(rmsle(xgb_optimal)))


# RandomForest with Optuna hyperparameter optimization framework
from sklearn.ensemble import RandomForestRegressor
def rf_objective(trial):
    n_estimators = int(trial.suggest_loguniform('n_estimators', 100, 300))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 15))
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['auto','sqrt','log2'])
    bootstrap = trial.suggest_categorical('bootstrap', ['True','False'])
    model = RandomForestRegressor(n_estimators=n_estimators, 
                          max_depth=max_depth,
                          min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf,
                          max_features=max_features,
                          bootstrap=bootstrap
                          )
    scores = cross_validate(model, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(rf_objective, n_trials=20)
best_trial_rf =  study.best_trial
print('Random Forest Optuna Best Trial')
print('RMSLE: {}'.format(np.sqrt(best_trial_rf.value * -1)))
print('Best params: {}'.format(best_trial_rf.params))

# Creating the optimized RandomForestRegressor model
n_estimators = int(best_trial_rf.params['n_estimators'])
max_depth = int(best_trial_rf.params['max_depth'])
min_samples_split = best_trial_rf.params['min_samples_split']
min_samples_leaf = best_trial_rf.params['min_samples_leaf']
max_features = best_trial_rf.params['max_features']
bootstrap = best_trial_rf.params['bootstrap']
rf_optimal = RandomForestRegressor(n_estimators=n_estimators,
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf,
                     max_features=max_features,
                     bootstrap=bootstrap
                     )
print('Random Forest Optimal Model Performance')
print('RMSLE: {}'.format(rmsle(rf_optimal)))


# SVR with Optuna hyperparameter optimization framework
from sklearn.svm import SVR
def svr_objective(trial):
    degree = trial.suggest_int('degree', 1, 6)
    kernel = trial.suggest_categorical('kernel', ['rbf','linear','poly','sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale','auto'])
    C = trial.suggest_int('C', 700, 900)
    model = SVR(degree = degree, 
                kernel = kernel,
                gamma = gamma,
                C = C
                )
    scores = cross_validate(model, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(svr_objective, n_trials=20)
best_trial_svr =  study.best_trial
print('SVR Optuna Best Trial')
print('RMSLE: {}'.format(np.sqrt(best_trial_svr.value * -1)))
print('Best params: {}'.format(best_trial_svr.params))

# Creating the optimized SVR model
degree = int(best_trial_svr.params['degree'])
kernel = best_trial_svr.params['kernel']
gamma = best_trial_svr.params['gamma']
C = int(best_trial_svr.params['C'])
svr_optimal = SVR(degree = degree,
                  kernel = kernel,
                  gamma = gamma,
                  C = C
                  )
print('SVR Optimal Model Performance')
print('RMSLE: {}'.format(rmsle(svr_optimal)))


# SECTION 3: ENSEMBLE ALGORITHM DEVELOPMENT
# Option 1: Voting Regressor
from sklearn.ensemble import VotingRegressor
voting_reg = VotingRegressor([('cb',cb_optimal), ('xgb',xgb_optimal), ('rf',rf_optimal), ('svr',svr_optimal)])
print('Voting Regressor')
print('RMSLE: {}'.format(rmsle(voting_reg)))

# Option 2: Stacking Regressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
# Function for stacking ensemble
def get_stacking():
    # Base models
    level0 = list()
    level0.append(('cb', cb_optimal))
    level0.append(('xgb', xgb_optimal))
    level0.append(('rf', rf_optimal))
    level0.append(('svr', svr_optimal))
	# Meta learner model
    level1 = LinearRegression()
	# Stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=kfold)
    return model
# Create model list
def get_models():
    models = dict()
    models['cb'] = cb_optimal
    models['xgb'] = xgb_optimal
    models['rf'] = rf_optimal
    models['svr'] = svr_optimal
    models['stacking'] = get_stacking()
    return models
models = get_models()
# Evaluate the models and store their respective RMSLE's
results, names = list(), list()
for name, model in models.items():
    scores = rmsle(model)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# It seems that the Voting Regressor ensemble method yields a better RMSLE score, so let's use that to predict the final result


# SECTION 4: CALCULATE FINAL Y TEST VALUES
# Importing the final test dataset
test_dataset_with_id = pd.read_csv('test.csv')

# Dropping the ID column
test_dataset = test_dataset_with_id.iloc[:,1:]

# Convert MSSubClass to object data type (because it is categorical in nature despite having numbers)
test_dataset = test_dataset.astype({'MSSubClass':'object'})

# For numerical columns, fill NA values with 0
for column in [column for column in numerical_columns if column not in ['SalePrice']]:
    test_dataset[column].fillna(0, inplace=True)

# Dropping the chosen numerical and categorical columns
test_dataset_cleaned = test_dataset.drop(numerical_columns_to_drop + categorical_columns_to_drop, axis=1)

# Creating dummy variables for categorical columns
test_dataset_cleaned = pd.get_dummies(test_dataset_cleaned, columns=[column for column in categorical_columns if column not in categorical_columns_to_drop])

# Align test dataset with train X to ensure they have the same columns
X, test_dataset_cleaned = X.align(test_dataset_cleaned, join='left', axis=1)

# Replace NaN with 0's for columns which were present in train X but not in test
test_dataset_cleaned.fillna(0, inplace=True)

# Feature Scaling for numerical columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_dataset_cleaned_scaled = test_dataset_cleaned.copy()
numerical_column_count = len(numerical_columns) - len(numerical_columns_to_drop)
test_dataset_cleaned_scaled.iloc[:,:numerical_column_count-1] = sc.fit_transform(test_dataset_cleaned_scaled.iloc[:,:numerical_column_count-1])

# Predict final Y Test Values
voting_reg.fit(X_scaled, y)
test_y_pred = voting_reg.predict(test_dataset_cleaned_scaled)

# Turning final Y Test Values into dataframe and exporting to csv
test_y_pred_df = pd.DataFrame(test_y_pred, index=test_dataset_with_id['Id'], 
                    columns=['SalePrice'])
test_y_pred_df.to_csv('test_y_pred_vF2.csv')



