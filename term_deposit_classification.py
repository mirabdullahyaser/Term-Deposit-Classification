import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers.experimental import preprocessing
from xgboost import XGBClassifier

CSV_NAME = 'bank+marketing/bank-additional/bank-additional/bank-additional-full.csv'
OUTPUT_LABEL = 'y'
OUTPUT_LABEL_NAME = 'term_deposit'
TEST_SIZE = 0.2
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
DECISION_TREES_MAX_DEPTH = 5
SVM_C = 4
SVM_KERNEL = 'linear'
KNN_NEIGHBORS = 6
RANDOM_FOREST_ESTIMATORS = 160
GRADIENT_BOOSTING_ESTIMATORS = 170
ACTIVATATION_FUNCTION = "relu"
ACTIVATATION_FUNCTION_OUTPUT = "sigmoid"
LOSS_FUNCTION ="binary_crossentropy"
METRICS = 'accuracy'
LEARNING_RATE = 0.0001
EPOCHS = 20
VALIDATION_SPLIT = 0.2


def read_data():
    dataset = pd.read_csv(CSV_NAME, sep=';')
    labels = dataset.pop(OUTPUT_LABEL)
    features = dataset
    return features, labels

def data_preprocessing(features, labels):
    # labels = pd.get_dummies(labels.astype(str),prefix=OUTPUT_LABEL_NAME)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels.astype(str))
    features = pd.get_dummies(features, columns=CATEGORICAL_FEATURES,)
    return features, labels, label_encoder

def show_results(acc, loss, class_report, conf_matrix):
    print(f'Accuracy: {acc}')
    print(f'Log Loss: {loss}')
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

def logistic_regression(train_features, train_labels, test_features, test_labels):
    model = LogisticRegression().fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nLogistic Regression Classification Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def decision_tree_classifier(train_features, train_labels, test_features, test_labels):
    model = DecisionTreeClassifier(max_depth=DECISION_TREES_MAX_DEPTH).fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nDecision Tree Classifier Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def svm(train_features, train_labels, test_features, test_labels):
    model = SVC(kernel=SVM_KERNEL, C=SVM_C).fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nSupport Vector Machine Classification Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def naive_bayes_classifier(train_features, train_labels, test_features, test_labels):
    model = GaussianNB().fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nNiave Bayes Classification Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def knn(train_features, train_labels, test_features, test_labels):
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS).fit(train_features,train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nK Nearest Neighbors Classification Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def random_forest_classifier(train_features, train_labels, test_features, test_labels):
    model = RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS, random_state=42).fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nRandom Forest Classifier Evaluation")
    show_results(acc, loss, class_report, conf_matrix)

def gradient_boosting_classifier(train_features, train_labels, test_features, test_labels):
    model = GradientBoostingClassifier(n_estimators=GRADIENT_BOOSTING_ESTIMATORS, random_state=42).fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nGradient Boosting Classifier Evaluation")
    show_results(acc, loss, class_report, conf_matrix)
    
def xgboost_classifier(train_features, train_labels, test_features, test_labels):
    model = XGBClassifier().fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels)
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nXGBoost Classifier Evaluation")
    show_results(acc, loss, class_report, conf_matrix)
    
def dnn_model(train_labels, train_features, test_labels, test_features):
    train_features = train_features.astype('float32')
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    model = tf.keras.models.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(12, activation=ACTIVATATION_FUNCTION))
    model.add(tf.keras.layers.Dense(8, activation=ACTIVATATION_FUNCTION))
    model.add(tf.keras.layers.Dense(4, activation=ACTIVATATION_FUNCTION))
    model.add(tf.keras.layers.Dense(1, activation = ACTIVATATION_FUNCTION_OUTPUT))
    model.compile(loss=LOSS_FUNCTION,optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE),metrics=[METRICS])
    model_history = model.fit(train_features, train_labels, validation_split=VALIDATION_SPLIT, verbose=2, epochs=EPOCHS)

def boot():
    features, labels = read_data()
    #
    features, labels, label_encoder = data_preprocessing(features, labels)
    #
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=TEST_SIZE, random_state=42)
    #
    # decision_tree_classifier(train_features, train_labels, test_features, test_labels)
    # svm(train_features, train_labels, test_features, test_labels)
    # logistic_regression(train_features, train_labels, test_features, test_labels)
    # knn(train_features, train_labels, test_features, test_labels)
    # random_forest_classifier(train_features, train_labels, test_features, test_labels)
    # gradient_boosting_classifier(train_features, train_labels, test_features, test_labels)
    # xgboost_classifier(train_features, train_labels, test_features, test_labels)
    dnn_model(train_labels, train_features, test_labels, test_features)
    

if __name__ == '__main__':
    boot()
    