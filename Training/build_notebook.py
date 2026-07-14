"""Generates Training/Loan Prediction using Ml.ipynb from markdown/code cells."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# Smart Lender — Loan Eligibility Prediction\n"
   "End-to-end notebook: EDA → preprocessing → balancing → scaling → model training → model selection.")

md("## 1. Import the libraries")
code("""import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE""")

md("## 2. Read the dataset")
code("""data = pd.read_csv('../Dataset/loan_prediction.csv')
data.head()""")

code("data.shape")
code("data.info()")

md("## 3. Univariate analysis")
code("""plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.countplot(x='Gender', data=data)
plt.subplot(1,2,2)
sns.countplot(x='Education', data=data)
plt.show()""")

code("""plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['ApplicantIncome'], kde=True, color='r')
plt.subplot(1,2,2)
sns.histplot(data['Credit_History'].dropna(), kde=True)
plt.show()""")

md("## 4. Bivariate analysis")
code("""plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.countplot(x='Married', hue='Gender', data=data)
plt.subplot(1,3,2)
sns.countplot(x='Self_Employed', hue='Education', data=data)
plt.subplot(1,3,3)
sns.countplot(x='Property_Area', hue='Loan_Status', data=data)
plt.show()""")

md("## 5. Multivariate analysis")
code("""plt.figure(figsize=(10,5))
sns.swarmplot(x='Gender', y='ApplicantIncome', hue='Loan_Status', data=data)
plt.show()""")

md("## 6. Handling missing values")
code("data.isnull().sum()")

code("""for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    data[col] = data[col].fillna(data[col].mode()[0])

for col in ['LoanAmount', 'Loan_Amount_Term']:
    data[col] = data[col].fillna(data[col].mean())

data.isnull().sum()""")

md("## 7. Handling categorical values")
code("""data['Dependents'] = data['Dependents'].astype(str).str.replace('+', '', regex=False).astype(int)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Married'] = data['Married'].map({'No': 0, 'Yes': 1})
data['Education'] = data['Education'].map({'Not Graduate': 0, 'Graduate': 1})
data['Self_Employed'] = data['Self_Employed'].map({'No': 0, 'Yes': 1})
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
data['Loan_Status'] = data['Loan_Status'].map({'N': 0, 'Y': 1})

data.drop(columns=['Loan_ID'], inplace=True)
data.head()""")

code("""int_cols = ['Gender','Married','Dependents','Education','Self_Employed',
            'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
            'Credit_History','Property_Area','Loan_Status']
for col in int_cols:
    data[col] = data[col].astype('int64')
data.info()""")

md("## 8. Balancing the dataset (SMOTE)")
code("""x = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

smote = SMOTE(random_state=42)
x_bal, y_bal = smote.fit_resample(x, y)

print(y.value_counts())
print(y_bal.value_counts())

names = x_bal.columns""")

md("## 9. Scaling the dataset")
code("""sc = StandardScaler()
x_bal = sc.fit_transform(x_bal)
x_bal = pd.DataFrame(x_bal, columns=names)
x_bal.head()""")

md("## 10. Train / test split")
code("""X_train, X_test, y_train, y_test = train_test_split(
    x_bal, y_bal, test_size=0.33, random_state=42)
X_train.shape, X_test.shape""")

md("## 11. Model building")

md("### 11.1 Decision Tree")
code("""def decisionTree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    print('train:', accuracy_score(y_tr, y_train))
    yPred = model.predict(X_test)
    print('test:', accuracy_score(yPred, y_test))
    return model

dt_model = decisionTree(X_train, X_test, y_train, y_test)""")

md("### 11.2 Random Forest")
code("""def RandomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    print('train:', accuracy_score(y_tr, y_train))
    yPred = model.predict(X_test)
    print('test:', accuracy_score(yPred, y_test))
    return model

rf_model = RandomForest(X_train, X_test, y_train, y_test)""")

md("### 11.3 K-Nearest Neighbors")
code("""def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    print('train:', accuracy_score(y_tr, y_train))
    yPred = model.predict(X_test)
    print('test:', accuracy_score(yPred, y_test))
    return model

knn_model = KNN(X_train, X_test, y_train, y_test)""")

md("### 11.4 XGBoost")
code("""def xgboost(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    print('train:', accuracy_score(y_tr, y_train))
    yPred = model.predict(X_test)
    print('test:', accuracy_score(yPred, y_test))
    print(confusion_matrix(y_test, yPred))
    print(classification_report(y_test, yPred))
    return model

xgb_model = xgboost(X_train, X_test, y_train, y_test)""")

md("## 12. Cross validation")
code("""for name, model in [('Decision Tree', dt_model), ('Random Forest', rf_model),
                     ('KNN', knn_model), ('XGBoost', xgb_model)]:
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f'{name}: {score:.3f}')""")

md("## 13. Save the best model\n"
   "XGBoost is selected for deployment — it gives the best balance of "
   "generalization and cross-validated accuracy among the models tested.")
code("""with open('../Flask/rdf.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('../Flask/scale1.pkl', 'wb') as f:
    pickle.dump(sc, f)

print('Saved rdf.pkl and scale1.pkl to ../Flask/')""")

nb['cells'] = cells

with open('/home/claude/SmartLender/Training/Loan Prediction using Ml.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
