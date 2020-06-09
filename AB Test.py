import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('ab_data.csv')

print(df.head())

#Checking to see if there are any users in control group that saw new page and users in treatment group that saw old page
print(df.groupby(['group','landing_page']).count())

# Removing control/new_page and treatment/old_page
df_cleaned = df.loc[(df['group'] == 'control') & (df['landing_page'] == 'old_page') | (df['group'] == 'treatment') & (df['landing_page'] == 'new_page')]
print(df_cleaned.groupby(['group','landing_page']).count())

groups = df_cleaned.groupby(['group','landing_page','converted']).size()
groups.plot.bar()
plt.show()

df_cleaned['landing_page'].value_counts().plot.pie()
plt.show()

# Re-arrrange data into 2x2 for Chi-Squared

a = df_cleaned[df_cleaned['group'] == 'control']
b = df_cleaned[df_cleaned['group'] == 'treatment']

a_click = a.converted.sum()
a_noclick = a.converted.size - a.converted.sum()
b_click = b.converted.sum()
b_noclick = b.converted.size - b.converted.sum()

T = np.array([[a_click, a_noclick], [b_click, b_noclick]])

print(scipy.stats.chi2_contingency(T,correction=False)[1])

# Conversion rates
a_CTR = a_click / (a_click + a_noclick)
b_CTR = b_click / (b_click + b_noclick)
print(a_CTR, b_CTR)

#There is no significance in conversions between the old and new webpage