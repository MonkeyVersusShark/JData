import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

df = pd.read_csv('train_feat.csv', index_col=0)
y = df.label
X = df.drop(['uid', 'label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def evaluate_classifier(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    from sklearn.externals import joblib

    classifier = XGBClassifier(
                    learning_rate =0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred = y_pred[:, np.newaxis]
    F1 = f1_score(y_test, y_pred)
    auc = accuracy_score(y_test, y_pred)
    score = 0.6 * auc + 0.4 * F1
    joblib.dump(classifier, "train_model.m")
    print(classification_report(y_test, y_pred))
    print(score)

evaluate_classifier(X_train, X_test, y_train, y_test)