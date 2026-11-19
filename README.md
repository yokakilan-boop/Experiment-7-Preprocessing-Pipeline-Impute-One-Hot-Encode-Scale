# Preprocessing-Pipeline-Impute-One-Hot-Encode-Scale
## EXPERIMENT - 7

## AIM

Build a ColumnTransformer pipeline to preprocess numeric and categorical features 
safely. 

---

## ALGORITHM

**Brief Algorithm:**

Apply median imputation & standard scaling to numeric features; one-hot encode 
categorical features — all inside a pipeline to avoid leakage. 

---

## PROGRAM

**Run the following program in a Jupyter Notebook code cell:**

```python
import pandas as pd

from sklearn.compose import ColumnTransformer 

from sklearn.pipeline import Pipeline 

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import OneHotEncoder, StandardScaler 

from sklearn.ensemble import RandomForestClassifier 

df = pd.DataFrame({'age':[25,35, None,45], 'income':[50,40,30,60], 'city':['X', 'Y', 'X', 'Z'], 'target':[0,1,
0,1]}) 

X = df[['age', 'income', 'city']]; y = df['target']

numeric_features = ['age', 'income'] 

numeric_transformer = Pipeline([('imputer', 

SimpleImputer(strategy='median')), ('scaler', StandardScaler())]) 

cat_features = ['city']; cat_transformer = Pipeline([('onehot', 

OneHotEncoder(handle_unknown='ignore'))]) 

preprocessor = ColumnTransformer([('num', numeric_transformer, 

numeric_features), ('cat', cat_transformer, cat_features)]) 

clf = Pipeline([('pre', preprocessor), ('rf', 

RandomForestClassifier(random_state=42))]) 

clf.fit(X,y)

print('Preprocessor output shape:',

preprocessor.transform(X).toarray().shape)

print('Pipeline trained successfully.') 
```

---

## DETAILED EXPLANATION

1. Create toy DataFrame with a missing age. 
2. Numeric pipeline: SimpleImputer(strategy='median') then StandardScaler(); 
fit only on training data when used in cross-validation. 
3. Categorical pipeline: OneHotEncoder(handle_unknown='ignore') to handle 
unseen categories at transform time. 
4. ColumnTransformer applies the appropriate pipeline to each column subset. 
5. Pipeline chains preprocessing and classifier for reproducible workflow. 
6. preprocessor.transform(X).toarray() shows the processed numeric matrix fed 
into the classifier. 
---

---

## OUTPUT

```text
Preprocessor output shape: (4,5) 
Pipeline trained successfully.  
```

---

## RESULT

Students learn to encapsulate preprocessing to avoid leakage and ensure correct 
application during training and prediction.

---

## THEORY  

Pipelines ensure transformations are fit on training data only; critical for valid evaluation.
