from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=1))  # Sets undefined precision to 1.0

joblib.dump(svm, 'svm_model.pkl')