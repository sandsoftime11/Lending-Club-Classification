import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def fill_mort_acc(total_acc, mort_acc,data):
    total_acc_avg = data.groupby(by='total_acc').mean().mort_acc
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc

def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        st.write(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        st.write("_______________________________________________")
        st.write(f"CLASSIFICATION REPORT:\n{clf_report}")
        st.write("_______________________________________________")
        st.write(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        st.write("Test Result:\n================================================")        
        st.write(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        st.write("_______________________________________________")
        st.write(f"CLASSIFICATION REPORT:\n{clf_report}")
        st.write("_______________________________________________")
        st.write(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")        


def main():
    st.title("Lending Club Loan Default Prediction")
    st.sidebar.title("Lending Club Loan Default Prediction")
    st.markdown("Is the applicant is likely to repay the loan ? 🏦")
    st.sidebar.markdown("Is the applicant is likely to repay the loans? 🏦")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("lending_club_loan_two.csv")
        data=data.head(1000)

        data.drop('emp_title', axis=1, inplace=True)
        data.drop('emp_length', axis=1, inplace=True)
        data.drop('title', axis=1, inplace=True)
        data['mort_acc'] = data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc'],data), axis=1)
        data.dropna(inplace=True)


        term_values = {' 36 months': 36, ' 60 months': 60}
        data['term'] = data.term.map(term_values)

        data.drop('grade', axis=1, inplace=True)
        dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status', 
           'application_type', 'home_ownership']
        data = pd.get_dummies(data, columns=dummies, drop_first=True)


        data['zip_code'] = data.address.apply(lambda x: x[-5:])
        data = pd.get_dummies(data, columns=['zip_code'], drop_first=True)
        data.drop('address', axis=1, inplace=True)
        data.drop('issue_d', axis=1, inplace=True)
        data['earliest_cr_line'] = data.earliest_cr_line.str.split('-', expand=True)[1]
        

        data['loan_status'] = data.loan_status.map({'Fully Paid':0, 'Charged Off':1})
        return data
    
    @st.cache(persist=True)
    def split(df):
        y = df.loan_status
        x = df.drop(columns=['loan_status'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            
            st.subheader("Confusion Matrix")
            #plot_confusion_matrix(model, x_test, y_test,cmap='Blues', values_format='d', display_labels=["Charged Off","Fully Paid"])
            #st.pyplot()
            plot_confusion_matrix(model, x_test, y_test, cmap='Blues', values_format='d', display_labels=['Fully-Paid', 'Default'])
            st.pyplot()


        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ['Fully Paid','Default']
    
    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Random Forest","XGBoost","Artficial Neural Network"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


    if classifier == 'XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("XGBoost Results")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            model = XGBClassifier()

            x_train = np.array(x_train).astype(np.float32)
            x_test = np.array(x_test).astype(np.float32)
            y_train = np.array(y_train).astype(np.float32)
            y_test = np.array(y_test).astype(np.float32)

            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_train)
            y_pred = model.predict(x_test)


            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))

            plot_metrics(metrics)    


    if classifier == 'Artficial Neural Network':
        st.sidebar.subheader("Model Hyperparameters")
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Artficial Neural Network Results")
            st.set_option('deprecation.showPyplotGlobalUse', False)

            x_train = np.array(x_train).astype(np.float32)
            x_test = np.array(x_test).astype(np.float32)
            y_train = np.array(y_train).astype(np.float32)
            y_test = np.array(y_test).astype(np.float32)

            model = Sequential()

            model.add(Dense(x_train.shape[1], activation='relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(128, activation='relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(56, activation='relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(28, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])

            r = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test), 
                epochs=25, 
                batch_size=8,
            #     class_weight={0:w_n, 1:w_p}
            )      

            training_score = model.evaluate(x_train, y_train)
            testing_score = model.evaluate(x_test, y_test)

            print(f"TRAINING SCORE: {training_score}")
            print(f"TESTING SCORE: {testing_score}")                 

            plot_metrics(metrics)                   

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Lending Club Data Set (Classification)")
        st.write(df)
        st.markdown("LendingClub is a peer-to-peer lending platform which deals in lending unsecured personal and business loans to borrowers.  The dataset we seek to analyze consists of borrower data such as the loan amount, loan term, interest rates and related information.Our purpose with the present analysis is to develop a machine learning algorithm to predict loan defaulters based on certain variables contained in the dataset. Our objective is to accurately identify defaulters to assist the portfolio and risk assessment mechanism employed by LendingClub so that better judgement is exercised in respect of future credit sanctions.")

if __name__ == '__main__':
    main()

