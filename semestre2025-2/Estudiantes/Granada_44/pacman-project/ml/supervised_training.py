from feature_cleaning import clean_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#Train model
X = clean_df.drop(columns=["Action"])
y = clean_df["Action"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(X_train, y_train)

#Save model
with open('ml/trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained!!")