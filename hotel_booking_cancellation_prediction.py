import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("H2.csv")

# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=12, car_th=167):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat
cols_to_remove = ["ReservationStatusDate","ReservationStatus", "Agent", "Company","ArrivalDateYear","Babies"]
df.drop(cols_to_remove, axis=1, inplace=True)
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
df_yedek = df

#Exploratory data analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
for col in num_cols:
    num_summary(df, col)

#Explore variables based on target variable
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, "IsCanceled", col)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df, "IsCanceled", col)
def draw_boxplots_by_target_plotly(df, num_cols, target_var):
    """
    Draws interactive boxplots for numerical columns based on a target variable using Plotly.

    Parameters:
    - df: DataFrame containing the data.
    - num_cols: List of numerical columns to plot.
    - target_var: Target variable (categorical) to group by.

    Returns:
    None (shows the plots).
    """
    for col in num_cols:
        fig = px.box(
            df,
            x=target_var,
            y=col,
            title=f"Boxplot of {col} by {target_var}",
            labels={target_var: target_var, col: col},
            template="plotly_white",
            color=target_var
        )
        fig.show(renderer="browser")
draw_boxplots_by_target_plotly(df, num_cols, "IsCanceled")
def add_new_vars_to_df(df):
    # Transform PreviousCancellations to binary
    df['PreviousCancellations'] = df['PreviousCancellations'].apply(lambda x: 0 if x == 0 else 1)

    # Create NEW_RoomTypeMismatch
    df['NEW_RoomTypeMismatch'] = df.apply(lambda row: 1 if row['ReservedRoomType'] != row['AssignedRoomType'] else 0,
                                          axis=1)

    # Strip spaces from ReservedRoomType and AssignedRoomType
    df['ReservedRoomType'] = df['ReservedRoomType'].str.strip()
    df['AssignedRoomType'] = df['AssignedRoomType'].str.strip()

    # Create NEW_RoomTypeDowngraded
    room_order = {'A': 8, 'B': 7, 'C': 6, 'D': 5, 'E': 4, 'F': 3, 'G': 2, 'P': 1, 'K': 0}
    df['NEW_RoomTypeDowngraded'] = df.apply(
        lambda row: 1 if room_order[row['ReservedRoomType']] > room_order[row['AssignedRoomType']] else 0,
        axis=1
    )

    # Create NEW_ADR_by_Adults
    df["NEW_ADR_by_Adults"] = np.where(
        (df["Adults"] == 0) | (df["ADR"] == 0),
        0,
        (df["Adults"] + 1) / (df["ADR"] + 1)
    )

    # Create NEW_ADR_by_Children
    df["NEW_ADR_by_Children"] = np.where(
        (df["Children"] == 0) | (df["ADR"] == 0),
        0,
        (df["Children"] + 1) / (df["ADR"] + 1)
    )

    # Create NEW_ADR_by_LeadTime
    df["NEW_ADR_by_LeadTime"] = np.where(
        (df["LeadTime"] == 0),
        0,
        (df["ADR"] + 1) / (df["LeadTime"] + 1)
    )

    # Create NEW_Weeknights_by_LeadTime
    df["NEW_Weeknights_by_LeadTime"] = np.where(
        (df["LeadTime"] == 0),
        0,
        df["StaysInWeekNights"] / (df["LeadTime"] + 1)
    )
    df["TotalOfSpecialRequests"] = np.where(
        df["TotalOfSpecialRequests"] >= 3,  # Condition for "3 or more"
        "3 or more",  # Assign "3 or more" if condition is True
        df["TotalOfSpecialRequests"]  # Otherwise, keep the original value
    )

    # Map months to quarters and drop ArrivalDateMonth
    month_to_quarter = {
        "January": "Q1", "February": "Q1", "March": "Q1",
        "April": "Q2", "May": "Q2", "June": "Q2",
        "July": "Q3", "August": "Q3", "September": "Q3",
        "October": "Q4", "November": "Q4", "December": "Q4"
    }
    df["ArrivalDateQuarter"] = df["ArrivalDateMonth"].map(month_to_quarter)
    df.drop("ArrivalDateMonth", axis=1, inplace=True)


    # Filter rows where Adults != 0 or ADR != 0
    df = df[(df["Adults"] != 0) & (df["ADR"] != 0)]
    print(df.head())
    return df

add_new_vars_to_df(df)
df_yedek.head()
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "IsCanceled", cat_cols)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
df = rare_encoder(df, 0.02)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
num_but_cat = [col for col in num_but_cat if col not in ["IsCanceled", "TotalOfSpecialRequests"] ]
def combine_categorical_classes(df, cat_cols):
    """
    Combines the classes of categorical variables with numerical values into:
    - 0: Remains as 0.
    - 1: Remains as 1.
    - 2 or more: Collapsed into a single category labeled as '2 or more'.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - cat_cols (list): List of categorical columns to modify.

    Returns:
    - pd.DataFrame: The modified DataFrame with updated categorical columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_combined = df.copy()

    for col in cat_cols:
        # Apply transformation to combine classes
        df_combined[col] = df_combined[col].apply(lambda x: "0" if x == 0 else "1" if x == 1 else "2 or more")

    return df_combined

df = combine_categorical_classes(df,num_but_cat)
def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Outlier analysis and handling
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
cat_summary(df, "Country")

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if "IsCanceled" not in col]
dff = one_hot_encoder(df, cat_cols, drop_first=True)

dff = dff.sample(30000, random_state=42)
dff.to_csv("dff_30k.csv",index=False)

#dff = pd.read_csv("dff_30k.csv")

y = dff["IsCanceled"]
X = dff.drop(["IsCanceled"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          #('SVM', SVC(gamma='auto', random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345))
          #("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))
               ]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

########## LR ##########
#Accuracy: 0.8122
#Auc: 0.8937
#Recall: 0.6898
#Precision: 0.8301
#F1: 0.7535
########## CART ##########
#Accuracy: 0.8087
#Auc: 0.8063
#Recall: 0.7813
#Precision: 0.7645
#F1: 0.7727
########## RF ##########
#Accuracy: 0.857
#Auc: 0.9338
#Recall: 0.7648
#Precision: 0.8762
#F1: 0.8166
########## XGB ##########
#Accuracy: 0.8543
#Auc: 0.9319
#Recall: 0.7918
#Precision: 0.8482
#F1: 0.819
########## LightGBM ##########
#Accuracy: 0.8558
#Auc: 0.9335
#Recall: 0.7793
#Precision: 0.8613
#F1: 0.8182

rf_model = RandomForestClassifier(random_state=17)


rf_params = {"max_depth": [ None],
             "max_features": [  7, "sqrt"],
             "min_samples_split": [ 2, 5, 10],
             "n_estimators": [500, 700, 1000]}

#rf_best_params = {'max_depth': [None],
 #'max_features': ['sqrt'],
 #'min_samples_split': [2],
 #'n_estimators': [700]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

#joblib.dump(rf_final, "rf_model_30.joblib")


df.info()
cv_results = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc","recall","precision"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block = True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [ 0.1, 0.01, 0.05],
                  "max_depth": [15, 20, "None"],
                  "n_estimators": [8000, 1000, 1200],
                  "colsample_bytree": [0.3 ,0.5, 0.7]}

#xgboost_params_best_params = {'colsample_bytree': [0.5],
 #'learning_rate': [0.01],
 #'max_depth': [15],
 #'n_estimators': [1000]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc","recall","precision"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()

plot_importance(xgboost_final, X)



################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [ 500, 1000],
               "colsample_bytree": [ 0.7, 1]}

#lgbm_params_best_params_ = {'colsample_bytree': [0.7], 'learning_rate': [0.05], 'n_estimators': [2000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc","recall","precision"])
cv_results['test_accuracy'].mean()
cv_results['test_recall'].mean()
cv_results['test_precision'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

df.head()


plot_importance(lgbm_final, X)

################################################
# KNN
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc","recall","precision"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_recall'].mean()
cv_results['test_precision'].mean()

################################################
# CART
################################################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc","recall","precision"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_recall'].mean()
cv_results['test_precision'].mean()


cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)


cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc","recall","precision"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_recall'].mean()
cv_results['test_precision'].mean()


###### Looking for optimal classification threshold for better recall or f1 values - for RF model

# Cross-validation predictions with probabilities
cv_probs = cross_val_predict(
    rf_final,  # Trained model
    X,         # Full dataset (no split)
    y,         # Target variable
    cv=5,      # Number of folds
    method="predict_proba",  # Get probabilities
    n_jobs=-1
)[:, 1]  # Select probabilities for the positive class

thresholds = np.arange(0.0, 1.05, 0.05)

# Initialize lists for metrics
recalls = []
f1_scores = []

for thresh in thresholds:
    # Convert probabilities to binary predictions based on the threshold
    y_pred_thresh = (cv_probs >= thresh).astype(int)

    # Calculate recall and F1-score
    recalls.append(recall_score(y, y_pred_thresh))
    f1_scores.append(f1_score(y, y_pred_thresh))

# Find optimal thresholds
optimal_recall_threshold = thresholds[np.argmax(recalls)]
optimal_f1_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal Threshold for Recall: {optimal_recall_threshold:.2f}")
print(f"Recall at Optimal Threshold: {max(recalls):.2f}")

print(f"Optimal Threshold for F1-Score: {optimal_f1_threshold:.2f}")
print(f"F1-Score at Optimal Threshold: {max(f1_scores):.2f}")

# Plot Recall and F1-Score vs Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, recalls, label="Recall", marker="o")
plt.plot(thresholds, f1_scores, label="F1-Score", marker="x")
plt.axvline(optimal_recall_threshold, color="green", linestyle="--", label=f"Optimal Recall Threshold ({optimal_recall_threshold:.2f})")
plt.axvline(optimal_f1_threshold, color="red", linestyle="--", label=f"Optimal F1 Threshold ({optimal_f1_threshold:.2f})")
plt.title("Recall and F1-Score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()




