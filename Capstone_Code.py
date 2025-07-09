#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 10:09:54 2025

@author: liviamorales
"""

import numpy as np
import random
import scipy

from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

#seed with n-number 

n_number = "N12392083"
seed_value = int(n_number[1:])
np.random.seed(seed_value)
random.seed(seed_value)
print(f"Random Number Generator sseeded with: {seed_value}")
#load data
data = np.genfromtxt('rmpCapstoneNum.csv', delimiter=',')

column_averages = np.nanmean(data, axis = 0)
missing_data = np.isnan(data).sum(axis=0)

print(f"Average for each column: {column_averages}")
print(f"Number of missing values for each column: {missing_data}")

import pandas as pd
num_df = pd.read_csv('rmpCapstoneNum.csv', header = None)
num_df.columns = ['Average_Rating', 'Average_Difficulty', 'Number_of_Ratings', 'Pepper', 'Would_Take_Again', 'Online_Ratings', 'Male', 'Female']

column_averages = num_df.mean(numeric_only = True)

missing_data = num_df.isnull().sum()
print("Average for each column:")
print(column_averages)

print("\nNumber of missing values for each column:")
print(missing_data)

average_rating = 0
average_difficulty = 1
num_ratings = 2
pepper = 3
take_again = 4
online = 5
male = 6
female = 7

threshold = 5
#%%
#Question 1

valid_rows = ~np.isnan(data[:, average_rating]) & ~np.isnan(data[:, num_ratings])
valid_rows_2 = ~np.isnan(data[:, [average_rating, average_difficulty, num_ratings,take_again, pepper, online]])
data_clean_2 = data[valid_rows_2.all(axis = 1)]
data_clean_2 = data_clean_2[data_clean_2[:, num_ratings]>= threshold]

gender_filter = (data_clean_2[:, male] +data_clean_2[:, female]) == 1
data_clean_2 = data_clean_2[gender_filter]

#data to perform OLS
Y = data_clean_2[:, average_rating]
X = np.column_stack((
    data_clean_2[:, male],              
    data_clean_2[:, average_difficulty],
    data_clean_2[:, num_ratings],
    data_clean_2[:, take_again],
    data_clean_2[:, pepper],
    data_clean_2[:, online]
))

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())
print(model.pvalues)
p_value_male = model.pvalues[1]

print(f"P-value for male variable: {p_value_male:.15f}")
data_clean = data[valid_rows]

data_clean = data_clean[data_clean[:, num_ratings]>= threshold]

male_professors = data_clean[data_clean[:, male] == 1]
female_professors = data_clean[data_clean[:, female]== 1]

male_ratings = male_professors[:, average_rating]
female_ratings = female_professors[:, average_rating]

statistic, p_value = mannwhitneyu(male_ratings, female_ratings, alternative = 'greater')

print(f"Male Median Rating: {np.median(male_ratings): .2f}")
print(f"Female Median Rating: {np.median(female_ratings):.2f}")
print(f"Mann-Whitney U Statistic: {statistic:.2f}")
print(f"P-Value: {p_value:.5f}")

if p_value < 0.005:
    print(f"Result: Statistically signficant evidence of pro-male bias (p< 0.005).")
else:
    print(f"Result: No statistically significant evidence of pro-male bias.")

male_median = np.median(male_ratings)
female_median = np.median(female_ratings)

plt.figure(figsize=(8, 6))

box = plt.boxplot([male_ratings, female_ratings], labels=['Male', 'Female'], patch_artist=True,
                  medianprops=dict(color='red', linewidth=2))

# Fill the boxes with color
colors = ['lightblue', 'lightpink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.text(1, male_median + 0.05, f'Median: {male_median:.2f}', horizontalalignment='center', fontsize=5)
plt.text(2, female_median + 0.05, f'Median: {female_median:.2f}', horizontalalignment='center', fontsize=5)

plt.ylabel('Average Rating')
plt.title('Professor Ratings by Gender')
plt.grid(linestyle='--', alpha=0.7)
plt.show()

#%%
# question 2

valid_rows_q2 = ~np.isnan(data[:, [average_rating, num_ratings]])
data_clean_q2 = data[valid_rows_q2.all(axis=1)]

data_clean_q2 = data_clean_q2[data_clean_q2[:, num_ratings] >= 5]
                              # ) & (data_clean_q2[:, num_ratings] <= 150)

average_ratings_q2 = data_clean_q2[:, average_rating]
number_of_ratings_q2 = data_clean_q2[:, num_ratings]

corr, p_value = spearmanr(number_of_ratings_q2, average_ratings_q2)

print(f"Spearman correlation: {corr: .4f}")
print(f"P-value: {p_value:.6f}")

plt.figure(figsize=(8, 5))
plt.scatter(number_of_ratings_q2, average_ratings_q2, alpha = 0.5)


plt.xlabel("Number of ratings")

plt.ylabel("Average Rating")
plt.title("Experience (Over 5 Ratings) vs. Teaching Quality")
plt.grid(True, alpha = 0.5)
plt.show()

#%%
# question 3

valid_rows_q3 = ~np.isnan(data[:, [average_rating, average_difficulty]])
data_clean_q3 = data[valid_rows_q3.all(axis=1)]

average_ratings_q3 = data_clean_q3[:, average_rating]
average_difficulty_q3 = data_clean_q3[:, average_difficulty]

corr, p_value = pearsonr(average_difficulty_q3, average_ratings_q3)

print(f"Pearson Correlation (r) for rating and difficulty: {corr:.4f}")
print(f"P-value: {p_value: .30f}")

X = sm.add_constant(average_difficulty_q3)
model = sm.OLS(average_ratings_q3, X).fit()
print(model.summary())

plt.figure(figsize=(8,5))
sns.regplot(x=average_difficulty_q3, y=average_ratings_q3, scatter_kws={'alpha':0.4, 's':10})
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Relationship Between Average Rating and Average Difficulty')
plt.grid(True, linestyle='--',  alpha=0.6)
plt.tight_layout()
plt.show()
#%%
#Question 4

valid_rows_q4 = ~np.isnan(data[:, [average_rating, num_ratings, online]])
data_clean_q4 = data[valid_rows_q4.all(axis=1)]

data_clean_q4 = data_clean_q4[data_clean_q4[:, num_ratings] >= 5]

average_ratings_q4 = data_clean_q4[:, average_rating]
total_ratings_q4 = data_clean_q4[:, num_ratings]
online_ratings_q4 = data_clean_q4[:, online]


online_ratio_q4 = online_ratings_q4 / total_ratings_q4

online_heavy = data_clean_q4[online_ratio_q4 >= 0.7]
traditional = data_clean_q4[online_ratio_q4 < 0.7]

ratings_online = online_heavy[:, average_rating]
ratings_traditional = traditional[:, average_rating]

n1 = len(ratings_online)
n2 = len(ratings_traditional)

stat, p = mannwhitneyu(ratings_online, ratings_traditional, alternative='two-sided')

mean_u = n1 * n2 / 2
std_u = sqrt(n1 * n2 * (n1 + n2 +1)/12)
z = (stat - mean_u)/ std_u
r = abs(z) / sqrt(n1 +n2)
print(f"U statistic = {stat:.2f}")
print(f"Z = {z:.3f}")
print(f"Effect size r = {r:.3f}")

print(f"\n ≥70% Online Ratings")
print(f"Online-Heavy Median Rating: {np.median(ratings_online):.2f}")
print(f"Traditional Median Rating: {np.median(ratings_traditional):.2f}")
print(f"Mann-Whitney U Statistic: {stat:.2f}")
print(f"P-Value: {p}")


online_heavy_50 = data_clean_q4[online_ratio_q4 >= 0.5]
traditional_50 = data_clean_q4[online_ratio_q4 < 0.5]

ratings_online_50 = online_heavy_50[:, average_rating]
ratings_traditional_50 = traditional_50[:, average_rating]

stat_50, p_50 = mannwhitneyu(ratings_online_50, ratings_traditional_50, alternative='two-sided')

n1_50 = len(ratings_online_50)
n2_50 = len(ratings_traditional_50)

mean_u_50 = n1_50 * n2_50 / 2
std_u_50 = sqrt(n1_50 * n2_50 * (n1_50 + n2_50 +1)/12)
z_50 = (stat_50 - mean_u_50)/ std_u_50
r_50 = abs(z_50) / sqrt(n1_50 +n2_50)
print(f"U 50 statistic = {stat_50:.2f}")
print(f"Z 50 = {z_50:.3f}")
print(f"Effect size r for 50 = {r_50:.3f}")

print(f"\n ≥50% Online Ratings")
print(f"Online-Heavy Median Rating - 50: {np.median(ratings_online_50):.2f}")
print(f"Traditional Median Rating - 50: {np.median(ratings_traditional_50):.2f}")
print(f"Mann-Whitney U Statistic: {stat_50:.2f}")
print(f"P-Value: {p_50}")

all_ratings = [ratings_online_50, ratings_traditional_50, ratings_online, ratings_traditional]
labels = ['Online ≥50%', 'Traditional <50%', 'Online ≥70%', 'Traditional <70%']
colors = ['skyblue', 'lightgreen', 'lightblue', 'lightyellow']

# Plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(all_ratings, patch_artist=True, labels=labels)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for i, group in enumerate(all_ratings, start=1):
    plt.text(i, np.median(group) + 0.05, f"{np.median(group):.2f}", ha='center', fontsize=8)

plt.ylabel('Average Rating')
plt.title('Average Ratings by Teaching Modality (Thresholds at 50% and 70%)')

plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error

# Predictor: Proportion of online ratings
X_q4 = online_ratio_q4.reshape(-1,1)
Y_q4 = average_ratings_q4

X_q4_with_const = sm.add_constant(X_q4)

model_q4 = sm.OLS(Y_q4, X_q4_with_const).fit()

preds = model_q4.predict(X_q4_with_const)
rmse_q4 = np.sqrt(mean_squared_error(Y_q4, preds))

print(model_q4.summary())
print(f"RMSE: {rmse_q4:.4f}")

# Ensure no missing values in any of these columns
valid_cols = [average_rating, average_difficulty, pepper, num_ratings, take_again, online]
valid_rows_multi = ~np.isnan(data[:, valid_cols])
data_multi = data[valid_rows_multi.all(axis=1)]

data_multi = data_multi[data_multi[:, num_ratings] >= 5]

Y_multi = data_multi[:, average_rating]
X_multi = np.column_stack([
    data_multi[:, average_difficulty],
    data_multi[:, pepper],
    data_multi[:, num_ratings],
    data_multi[:, take_again] / 100,
    data_multi[:, online]
])

X_multi_with_const = sm.add_constant(X_multi)
model_multi = sm.OLS(Y_multi, X_multi_with_const).fit()

pred_multi = model_multi.predict(X_multi_with_const)
rmse_multi = np.sqrt(np.mean((Y_multi - pred_multi)**2))

print(model_multi.summary())
print(f"RMSE: {rmse_multi:.4f}")


#%%
#Question 5

valid_rows_q5 = ~np.isnan(data[:, [average_rating,num_ratings, take_again]])
data_clean_q5 = data[valid_rows_q5.all(axis=1)]

data_clean_q5 = data_clean_q5[data_clean_q5[:, num_ratings]>=5]
average_ratings_q5 = data_clean_q5[:, average_rating]
prop_take_again_q5 = data_clean_q5[:, take_again] / 100

corr, p_value = spearmanr(prop_take_again_q5, average_ratings_q5)

print(f"Spearman correlation: {corr: .4f}")
print(f"P-value: {p_value: .20f}")

corr_p, p_value_p = pearsonr(prop_take_again_q5, average_ratings_q5)

print(f"Pearson correlation: {corr_p:.4f}")
print(f"P-value for Pearson: {p_value_p:.4f}")
print(f"Pearson r-squared: {corr_p**2:.4f}")

#OLS coefficents
intercept = 1.6600
slope = 2.9786
r_squared = 0.775
plt.figure(figsize=(8,5))

sns.regplot(x=prop_take_again_q5, y=average_ratings_q5, scatter_kws={'alpha':0.4, 's':10}, line_kws ={'color': 'red'})
plt.xlabel('Proportion Who Would Take Again')
plt.ylabel('Average Rating')
plt.title('Relationship Between Average Rating and Would Take Again Percentage')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

X = sm.add_constant(prop_take_again_q5)
model = sm.OLS(average_ratings_q5, X).fit()
print(model.summary())

#%%
# Question 6

valid_rows_q6 = ~np.isnan(data[:, [average_rating,num_ratings, pepper]])
data_clean_q6 = data[valid_rows_q6.all(axis=1)]

data_clean_q6 = data_clean_q6[data_clean_q6[:, num_ratings]>=5]

pepper_professors = data_clean_q6[data_clean_q6[:, pepper]==1]
no_pepper_professors = data_clean_q6[data_clean_q6[:, pepper]== 0]

pepper_ratings = pepper_professors[:, average_rating]
no_pepper_ratings = no_pepper_professors[:, average_rating]

stat, p = mannwhitneyu(pepper_ratings, no_pepper_ratings, alternative= 'two-sided')

print(f'Pepper Professors Median Rating: {np.median(pepper_ratings):.2f}')
print(f'No Pepper Professors Median Rating: {np.median(no_pepper_ratings):.2f}')
print(f"Mann-Whitney U Statistic: {stat:.2f}")
print(f"P-Value: {p:.15f}")

plt.figure(figsize=(7,5))
plt.boxplot([pepper_ratings, no_pepper_ratings], tick_labels=['Recieved a Pepper', 'Did not Recieve a Pepper'])
plt.ylabel('Average Rating')
plt.title('Average Ratings: Recieved a Pepper vs. Did Not Recieve a Pepper Professors')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(pepper_ratings, bins=20, alpha=0.7, label='Received a Pepper', color='red')
plt.hist(no_pepper_ratings, bins=20, alpha=0.7, label='Did not Receive a Pepper', color='green')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Ratings: Pepper vs No Pepper')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

mean1, mean2 = np.mean(pepper_ratings), np.mean(no_pepper_ratings)
std1, std2 = np.std(pepper_ratings, ddof=1), np.std(no_pepper_ratings, ddof=1)

n1, n2 = len(pepper_ratings), len(no_pepper_ratings)
pooled_std = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2))

# Cohen's d
cohen_d = (mean1 - mean2) / pooled_std
print(f"Cohen's d: {cohen_d:.4f}")
#%%
#Question 7

valid_rows_q7 = ~np.isnan(data[:, [average_rating,num_ratings, average_difficulty]])
data_clean_q7 = data[valid_rows_q7.all(axis=1)]

data_clean_q7 = data_clean_q7[data_clean_q7[:, num_ratings]>=5]

X_q7 = data_clean_q7[:, average_difficulty]
Y_q7 = data_clean_q7[:, average_rating]

X_with_constant = sm.add_constant(X_q7)
model = sm.OLS(Y_q7, X_with_constant).fit()

r_squared_q7 = model.rsquared

predictions = model.predict(X_with_constant)
residuals = Y_q7 - predictions
rmse = np.sqrt(np.mean(residuals**2))

print(model.summary())
print(f"R-squared: {r_squared_q7: .4f}")
print(f"RMSE: {rmse: .4f}")

plt.figure(figsize=(8,5))

plt.scatter(X_q7, Y_q7, alpha=0.3, s=10, label='Observed Data')

X_sorted = np.sort(X_q7)
X_sorted_with_const = sm.add_constant(X_sorted)
y_pred_line = model.predict(X_sorted_with_const)

plt.plot(X_sorted, y_pred_line, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Predicting Average Rating from Difficulty')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(1, 5)   # Since ratings are bounded
plt.show()

# Plot residuals
plt.figure(figsize=(7, 5))
plt.scatter(X_q7, residuals, alpha=0.3, s=10)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Average Difficulty')
plt.ylabel('Residuals')
plt.title('Residual Plot: Rating vs Difficulty')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#%%
#Question 8

valid_rows_q8 = ~np.isnan(data[:, [average_rating, average_difficulty, num_ratings, pepper, take_again, online, male]])
data_clean_q8 = data[valid_rows_q8.all(axis=1)]
data_clean_q8 = data_clean_q8[data_clean_q8[:, num_ratings] >= 5]

# 2. define y and x
Y_q8 = data_clean_q8[:, average_rating]

X_q8 = np.column_stack((
    data_clean_q8[:, average_difficulty],
    data_clean_q8[:, num_ratings],
    data_clean_q8[:, pepper],
    data_clean_q8[:, take_again] / 100,   
    data_clean_q8[:, online],
    data_clean_q8[:, male]           
))

X_q8 = sm.add_constant(X_q8)

model_q8 = sm.OLS(Y_q8, X_q8).fit()

predictions_q8 = model_q8.predict(X_q8)
residuals_q8 = Y_q8 - predictions_q8
rmse_q8 = np.sqrt(np.mean(residuals_q8**2))

print(model_q8.summary())
print(f"RMSE: {rmse_q8:.4f}")
print(f"R-squared: {model_q8.rsquared: .5f}")

predictor_df = pd.DataFrame({
    'Average_Difficulty': data_clean_q8[:, average_difficulty],
    'Number_of_Ratings': data_clean_q8[:, num_ratings],
    'Pepper': data_clean_q8[:, pepper],
    'Would_Take_Again': data_clean_q8[:, take_again] / 100,
    'Online_Ratings': data_clean_q8[:, online],
    'Male': data_clean_q8[:, male],
})

corr_matrix = predictor_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Predictors (Q8)')
plt.tight_layout()
plt.show()

# residual plot
plt.figure(figsize=(7,5))
plt.scatter(predictions_q8, residuals_q8, alpha=0.3, s=10)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Residual Plot: Full Model (Q8)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


X_raw = np.column_stack([
    data_clean_q8[:, average_difficulty],
    data_clean_q8[:, num_ratings],
    data_clean_q8[:, pepper],
    data_clean_q8[:, take_again] / 100,
    data_clean_q8[:, online],
    data_clean_q8[:, male]  
])

#scale data with z-scores
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

X_std_with_const = sm.add_constant(X_std)

#model with z-scored data
model_std = sm.OLS(Y_q8, X_std_with_const).fit()

print(model_std.summary())

#%% PCR for q8

zData_q8 = scipy.stats.zscore(X_raw)

pca = PCA()
pca.fit(zData_q8)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.transform(zData_q8)

print(eigVals)

covarExplained = eigVals / sum(eigVals) * 100
numVariables = X_raw.shape[1]  

#plot eiegen vals
plt.figure(figsize=(6, 4))
plt.bar(np.arange(1, numVariables + 1), eigVals, color='skyblue')
plt.axhline(y=1, color='red', linestyle='--', label='Kaiser Criteria Cutoff')
plt.xticks(np.arange(1, numVariables + 1))
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues for Professor Ratings')
plt.legend()
plt.show()


feature_names = ['Difficulty', 'Num_Ratings', 'Pepper', 'Take_Again', 'Online', 'Male']

plt.figure(figsize=(6, 4))
plt.bar(feature_names, loadings[2], color='blue')
plt.xlabel('Predictor Variable')
plt.ylabel('Loading on PC1')
plt.title('Loadings of Predictors on PC1')
plt.tight_layout()
plt.show()

X_raw = np.column_stack([
    data_clean_q8[:, average_difficulty],
    data_clean_q8[:, num_ratings],
    data_clean_q8[:, pepper],
    data_clean_q8[:, take_again] / 100,
    data_clean_q8[:, online],
    data_clean_q8[:, male]
])
Y_q8 = data_clean_q8[:, average_rating]

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

pca = PCA()
X_pca = pca.fit_transform(X_std)
eigenvalues = pca.explained_variance_

num_components = sum(eigenvalues > 1)
X_pcr = X_pca[:, :num_components]

X_pcr_with_const = sm.add_constant(X_pcr)
model_pcr = sm.OLS(Y_q8, X_pcr_with_const).fit()

predictions = model_pcr.predict(X_pcr_with_const)
rmse = np.sqrt(mean_squared_error(Y_q8, predictions))
r_squared = model_pcr.rsquared

print(model_pcr.summary())
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r_squared:.4f}")



#%%
#question 9 - logistic regression classification model 


valid_rows_q9 = ~np.isnan(data[:, [average_rating, pepper]])
data_clean_q9 = data[valid_rows_q9.all(axis=1)]

X_q9 = data_clean_q9[:, average_rating].reshape(-1,1)
y_q9 = data_clean_q9[:, pepper]

# check class balance
unique, counts = np.unique(y_q9, return_counts=True)
print(f"Class distribution (Pepper=1, No Pepper=0): {dict(zip(unique, counts))}")

X_train, X_test, y_train, y_test = train_test_split(X_q9, y_q9, test_size=0.3, random_state=seed_value, stratify=y_q9)

clf = LogisticRegression(class_weight='balanced', random_state=seed_value)
clf.fit(X_train, y_train)

y_probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_probs)
print(f"AUC-ROC: {auc:.4f}")

from sklearn.metrics import confusion_matrix

y_pred = (y_probs >= 0.5).astype(int)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Pepper', 'Pepper'],
            yticklabels=['No Pepper', 'Pepper'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Predicting Pepper from Average Rating')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
#question 10
valid_rows_q10 = ~np.isnan(data[:, [average_rating, average_difficulty, num_ratings, take_again, online, male, pepper]])
data_clean_q10 = data[valid_rows_q10.all(axis=1)]

data_clean_q10 = data_clean_q10[data_clean_q10[:, num_ratings] >= 5]

X_raw_q10 = np.column_stack((
    data_clean_q10[:, average_rating],
    data_clean_q10[:, average_difficulty],
    data_clean_q10[:, num_ratings],
    data_clean_q10[:, take_again] / 100,
    data_clean_q10[:, online],
    data_clean_q10[:, male]
))
y_q10 = data_clean_q10[:, pepper]

# z-score
scaler = StandardScaler()
X_std_q10 = scaler.fit_transform(X_raw_q10)

pca = PCA(n_components=3)
X_pca_q10 = pca.fit_transform(X_std_q10)

seed_value = 12392083  
X_train, X_test, y_train, y_test = train_test_split(X_pca_q10, y_q10, test_size=0.3, stratify=y_q10, random_state=seed_value)

clf = LogisticRegression(class_weight='balanced', random_state=seed_value)
clf.fit(X_train, y_train)

# predict probabilty and roc curve
y_probs = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_probs)
print(f"AUC-ROC: {auc:.4f}")

y_pred = (y_probs >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Pepper from PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plot heatmap for confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Pepper', 'Pepper'],
            yticklabels=['No Pepper', 'Pepper'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Pepper Prediction with PCA')
plt.tight_layout()
plt.show()


valid_rows_q10 = ~np.isnan(data[:, [average_rating, average_difficulty, num_ratings, take_again, online, male, pepper]])
data_clean_q10 = data[valid_rows_q10.all(axis=1)]

# define features and target
X_q10 = np.column_stack((
    data_clean_q10[:, average_rating],
    data_clean_q10[:, average_difficulty],
    data_clean_q10[:, num_ratings],
    data_clean_q10[:, take_again] / 100,
    data_clean_q10[:, online],
    data_clean_q10[:, male]
))

y_q10 = data_clean_q10[:, pepper]

X_train, X_test, y_train, y_test = train_test_split(X_q10, y_q10, test_size=0.3, random_state=seed_value, stratify=y_q10)

clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed_value)
clf.fit(X_train, y_train)

y_probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_probs)
print(f"AUC-ROC (All Factors): {auc:.4f}")

y_pred_all = (y_probs >= 0.5).astype(int)

cm_all = confusion_matrix(y_test, y_pred_all)
print("Confusion Matrix (All Factors):")
print(cm_all)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Pepper', 'Pepper'],
            yticklabels=['No Pepper', 'Pepper'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Pepper Prediction with All Factors')
plt.tight_layout()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Predicting Pepper from All Factors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
#Extra credit 
num_df = pd.read_csv("rmpCapstoneNum.csv", header=None)
qual_df = pd.read_csv("rmpCapstoneQual.csv", header=None)

num_df.columns = ['Average_Rating', 'Average_Difficulty', 'Number_of_Ratings', 'Pepper', 'Would_Take_Again',
                  'Online_Ratings', 'Male', 'Female']
qual_df.columns = ['Major', 'School', 'State']

merged_df = pd.concat([num_df, qual_df], axis=1)

stem_keywords = ['Engineering', 'Biology', 'Math', 'Computer', 'Physics', 'Chemistry', 'Statistics', 'Data']
liberal_arts_keywords = ['History', 'Philosophy', 'English', 'Sociology', 'Political Science', 'Art', 'Music', 'Theater', 'Writing']

def classify_major(major):
    major = str(major)
    if any(stem in major for stem in stem_keywords):
        return 'STEM'
    elif any(lib in major for lib in liberal_arts_keywords):
        return 'LiberalArts'
    else:
        return 'Other'

merged_df['Major_Category'] = merged_df['Major'].apply(classify_major)

filtered_df = merged_df[merged_df['Major_Category'].isin(['STEM', 'LiberalArts'])]
filtered_df = filtered_df.dropna(subset=['Average_Rating', 'Average_Difficulty'])

filtered_df['Liberal_Arts_Yes'] = (filtered_df['Major_Category'] == 'LiberalArts').astype(int)

X = filtered_df[['Liberal_Arts_Yes', 'Average_Difficulty']]
X = sm.add_constant(X)
Y = filtered_df['Average_Rating']

model = sm.OLS(Y, X).fit()
print(model.summary())

filtered_df['Predicted_Rating'] = model.predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_df, x='Average_Difficulty', y='Predicted_Rating',
                hue='Major_Category', alpha=0.4, s=15)
sns.lineplot(data=filtered_df, x='Average_Difficulty', y='Predicted_Rating',
             hue='Major_Category', legend=False)

plt.title('Predicted Ratings by Difficulty and Major Category')
plt.xlabel('Average Difficulty')
plt.ylabel('Predicted Average Rating')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

ratings_stem = filtered_df[filtered_df['Major_Category'] == 'STEM']['Average_Rating']
ratings_liberal = filtered_df[filtered_df['Major_Category'] == 'LiberalArts']['Average_Rating']

corr_df = filtered_df[['Average_Rating', 'Average_Difficulty', 'Liberal_Arts_Yes']]
corr_matrix = corr_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

stat, p = mannwhitneyu(ratings_stem, ratings_liberal, alternative='two-sided')

print(f"Mann-Whitney U statistic: {stat:.2f}")
print(f"P-value: {p:.5f}")
print(f"Median Rating (STEM): {ratings_stem.median():.2f}")
print(f"Median Rating (Liberal Arts): {ratings_liberal.median():.2f}")


