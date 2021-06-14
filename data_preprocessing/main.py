import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import StringIO

csv_data = \
        '''A,B,C,D
        1.0,2.0,3.0,4.0
        5.0,6.0,,8.0
        10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull())
print(df.isnull().sum())

# convert DataFrame to np array
print(df.values)

# drop rows with missing values
print(df.dropna(axis=0))

# drop colums with missing values
print(df.dropna(axis=1))

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN
print(df.dropna(how='all'))

# drop rows that have fewer than 4 real values
print(df.dropna(thresh=4))

# only drop rows where NaN appear in specific columns (here: 'C')
print(df.dropna(subset=['C']))

# interpolate missing values with the mean from it's column
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

# same using pandas
print(df.fillna(df.mean()))

# now let's try the same with categorical values
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['colors', 'size', 'price', 'classlabel']
print(df)

# add numerical mapping to ordinal feature
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

# inverse mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))

# encode class labels
class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# inverse mapping for class labels:
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

# now with sklearn
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))

# use sklearn to encode colors
X= df[['colors', 'size', 'price']].values
print(X)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# we have mapped nominal feature as ordinal - not good. Let's use OHE instead
X = df[['colors', 'size', 'price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# same but more elegant
X = df[['colors', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
print(c_transf.fit_transform(X).astype(float))

# same, but even better
pd.get_dummies(df[['price', 'colors', 'size']])

# OHE introduces multicollinearity that can be alleviated by dropping one
# of the dummy columns
pd.get_dummies(df[['price', 'colors', 'size']],
               drop_first=True)

# same with OneHotEncoder:
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1, 2])
])
c_transf.fit_transform(X).astype(float)

# encoding ordinal features
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
df.columns = ['colors', 'size', 'price', 'classlabel']
df['x > M'] = df['size'].apply(
        lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(
        lambda x: 1 if x == 'XL' else 0)
del df['size']
print(df)
