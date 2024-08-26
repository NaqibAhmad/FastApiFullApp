import pandas as pd
from pprint import pprint
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('D:/InternshipPractical/FastAPIFinal/backend/app/bank.csv', sep = ";")

pprint(df.head())
DropCols= ['age','marital','education', 'default', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
#experiment Name
mlflow.set_experiment("CreditScoring")

X= df.drop(DropCols, axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

categorical_cols= X.select_dtypes(include=['object']).columns
numerical_cols= X.select_dtypes(exclude=['object']).columns

print(categorical_cols)
print(numerical_cols)

with mlflow.start_run():
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('numerical', 'passthrough', numerical_cols)
        ]
    )
    model= RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=
        [('preprocessor', preprocessor),
          ('classifier', model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall=recall_score(y_test, y_pred, pos_label='yes')
    mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Precision", str(precision))
    mlflow.log_metric("Recall", str(recall))

    mlflow.sklearn.log_model(pipeline, "CreditScore_Final")

    