import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import copy
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
from tensorflow.keras import regularizers
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
ACTIVATION_FUNCTION = "relu"
ACTIVATION_FUNCTION_OUTPUT = "sigmoid"
LOSS_FUNCTION ="binary_crossentropy"
METRICS = 'accuracy'
LEARNING_RATE = 0.0001
EPOCHS = 100
VALIDATION_SPLIT = 0.2
L2_STRENTH = 0.01


def read_data():
    dataset_complete = pd.read_csv(CSV_NAME, sep=';')
    features = copy.deepcopy(dataset_complete)
    labels = features.pop(OUTPUT_LABEL)
    return dataset_complete, features, labels

def data_preprocessing(features, labels):
    features = features.drop('cons.conf.idx', axis=1)
    features = features.drop('campaign', axis=1)
    missing_values_features = features.isnull().sum()
    missing_values_labels = labels.isnull().sum()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels.astype(str))
    features = pd.get_dummies(features, columns=CATEGORICAL_FEATURES,)
    return features, labels, label_encoder

def show_results(acc, loss, class_report, conf_matrix):
    print(f'Accuracy: {acc}')
    print(f'Log Loss: {loss}')
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

def plot_label_distribution(labels):
    value_counts = labels.value_counts()
    fig = px.bar(x=value_counts.index, y=value_counts.values, text=value_counts.values,
                 labels={'x':'Classes', 'y':'Count'})
    fig.update_traces(marker_color=['red', 'blue'])
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='white',
        title_text='Distribution of Classes in Output Label',
        title_x=0.5
    )
    fig.write_image('label_distribution.png')
    
def correlation_plot(dataset):
    label_encoder = LabelEncoder()
    dataset['y'] = label_encoder.fit_transform(dataset['y'].astype(str))
    numeric_dataset = dataset.select_dtypes(include=['number'])
    df_corr = numeric_dataset.corr().round(2)
    fig = ff.create_annotated_heatmap(
            z=df_corr.to_numpy(),
            x=df_corr.columns.tolist(),
            y=df_corr.index.tolist(),
            zmax=1, zmin=-1,
            showscale=True,
            hoverongaps=True,
            colorscale='Viridis'
            )
    fig.update_layout(title='Correlation Matrix', autosize=True, title_x=0.5, margin=dict(l=65, r=50, b=65, t=120))
    fig.write_image('correlation_matrix.png')


def plot_confusion_matrix(confusion_matrix, class_names, title):
    z = confusion_matrix.astype(int)
    x = class_names
    y = class_names
    z_text = [[str(y) for y in x] for x in z]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(title_text='<i><b>'+title + ' Confusion matrix</b></i>', title_x=0.5)
    fig.add_annotation(dict(font=dict(color="black",size=14),x=0.5,y=-0.15,showarrow=False,text="Predicted value",xref="paper",yref="paper"))
    fig.add_annotation(dict(font=dict(color="black",size=14),x=-0.1,y=0.5,showarrow=False,text="Real value",textangle=-90,xref="paper",yref="paper"))
    fig.write_image('confusion_matrix.png')

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

def gradient_boosting_classifier(train_features, train_labels, test_features, test_labels, label_encoder):
    model = GradientBoostingClassifier(n_estimators=GRADIENT_BOOSTING_ESTIMATORS, random_state=42).fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = accuracy_score(pred, test_labels)
    conf_matrix = confusion_matrix(pred, test_labels, labels=[0, 1])
    class_report = classification_report(pred, test_labels, zero_division=1)
    loss = log_loss(pred, test_labels)
    print("\nGradient Boosting Classifier Evaluation")
    show_results(acc, loss, class_report, conf_matrix)
    plot_confusion_matrix(conf_matrix, list(label_encoder.inverse_transform([0, 1])), "Gradient Boosting")
    
    
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
    test_features = test_features.astype('float32')
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    model = tf.keras.models.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(12, activation=ACTIVATION_FUNCTION, kernel_regularizer=regularizers.l2(L2_STRENTH)))
    model.add(tf.keras.layers.Dense(8, activation=ACTIVATION_FUNCTION, kernel_regularizer=regularizers.l2(L2_STRENTH)))
    model.add(tf.keras.layers.Dense(4, activation=ACTIVATION_FUNCTION, kernel_regularizer=regularizers.l2(L2_STRENTH)))
    model.add(tf.keras.layers.Dense(1, activation=ACTIVATION_FUNCTION_OUTPUT, kernel_regularizer=regularizers.l2(L2_STRENTH)))
    model.compile(loss=LOSS_FUNCTION, optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE), metrics=[METRICS])
    model_history = model.fit(train_features, train_labels, validation_split=VALIDATION_SPLIT, verbose=2, epochs=EPOCHS)
    scores = model.evaluate(test_features, test_labels)
    # pred = model.predict(test_features)
    print("\nDeep Neural Network Evaluation")
    show_results(scores[1], scores[0], None, None)
    return model, model_history


def boot():
    dataset, features, labels = read_data()
    #
    correlation_plot(dataset)
    plot_label_distribution(labels)
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
    gradient_boosting_classifier(train_features, train_labels, test_features, test_labels, label_encoder)
    # xgboost_classifier(train_features, train_labels, test_features, test_labels)
    # dnn_model(train_labels, train_features, test_labels, test_features)
    

if __name__ == '__main__':
    boot()
    