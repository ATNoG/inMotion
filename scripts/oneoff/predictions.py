import joblib
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

df = pl.read_csv("data/processed/dataset.csv")
df = df.drop_nulls()

print(f"{df}")

X = df.select([f"{i}" for i in range(1, 11)])
y = df.select(["label"]).to_numpy().ravel()
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(f"Training Data : {len(X_train)}")
print(f"Testing Data  : {len(X_test)}")

# define the list of classifiers
clfs = [
    ("LR", LogisticRegression(random_state=42, solver="newton-cg", max_iter=3000)),
    ("KNN", KNeighborsClassifier(n_neighbors=1)),
    ("NB", GaussianNB()),
    ("RFC", RandomForestClassifier(random_state=42)),
    ("MLP", MLPClassifier(random_state=42, learning_rate="adaptive", max_iter=5000)),
]

# whenever possible used joblib to speed-up the training
with joblib.parallel_backend("loky", n_jobs=-1):
    for label, clf in clfs:
        # train the model
        clf.fit(X_train, y_train)

        # generate predictions
        predictions = clf.predict(X_test)

        # compute the performance metrics
        mcc = matthews_corrcoef(y_test, predictions)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        print(f"{label:3} {acc:.2f} {f1:.2f} {mcc:.2f}")
