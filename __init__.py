import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def data_summary(dataframe):
    """
    It returns the summary of data

    Parameters:
    ----------------
        dataframe: dataframe
                dataframe that wants to apply  
                
    Returns:
    ---------------
    shape:  number of observations  
    dtypes: types of the columns
    isnull: number of the null values of the each columns
    describe: descriptive statistical analysis of data
    """
    print("############## SHAPE ##############")
    print(dataframe.shape[0])
    print("############## TYPES ##############")
    print(dataframe.dtypes)
    print("############## NULL ##############")
    print(dataframe.isnull().sum())
    print("############ DESCRIBE ############")
    print(dataframe.describe([0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)



def evaluation(model, x_train, x_test, y_train, y_test):
    """
    It returns the  results of the models and evaluations

    Parameters:
    ----------------
        model: dict
                dictionary format for models that wants to apply
        x_train: array
                train set without target column
        y_train: array
                train target column
        x_test: array
                test set without target column
        y_test: array
                test target column
                
    Returns:
    ---------------
    df_model: Dataframe that include accuracy, precision, recall and F1 score
    """

    accuracy, precision, recall, f1 = {}, {}, {}, {}

    for key in model.keys():
        
        # Fit the model
        model[key].fit(x_train, y_train)
        
        # Make predictions
        predictions = model[key].predict(x_test.values)
        
        # Calculate metrics
        accuracy[key] = accuracy_score(y_test.values, predictions)
        precision[key] = precision_score(y_test.values, predictions)
        recall[key] = recall_score(y_test.values, predictions)
        f1[key] = f1_score(y_test.values, predictions)


    df_model = pd.DataFrame(index=model.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['F1 Score'] = f1.values()


    return df_model


def rf_best_parameters(X_train, X_test, y_train, y_test):
    """
     it is find best parameters for Random Forest Algorithm by using RandomizedSearchCV

    Parameters:
    ----------------
        x_train: array
                train set without target column
        y_train: array
                train target column
        x_test: array
                test set without target column
        y_test: array
                test target column
                
    Returns:
    ---------------
    rf_randomcv: best parameters
    """
    rf_params = {'bootstrap': [True],
        'max_depth': [80, 90, 100],
        'max_features': ['sqrt', 'log2'],
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy', 'log_loss']}
    rf = RandomForestClassifier()
    rf_randomcv = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=100, cv=5, scoring='accuracy').fit(X_train,y_train)
    print("Best training score: {} with parameters: {}".format(rf_randomcv.best_score_, rf_randomcv.best_params_))
    print()

    rf = RandomForestClassifier(**rf_randomcv.best_params_)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    
    accuracy_test = accuracy_score(y_test, y_pred)
    print("Test score: {}".format(accuracy_test))
    print()

    return rf_randomcv


def knn_best_paramaters(X_train, X_test, y_train, y_test):
        """
        it is find best parameters for KNN Algorithm by using RandomizedSearchCV

        Parameters:
        ----------------
        x_train: array
                train set without target column
        y_train: array
                train target column
        x_test: array
                test set without target column
        y_test: array
                test target column
                
        Returns:
        ---------------
        knn_randomcv: best parameters
        """
        knn_grid = { 'n_neighbors': list(range(1,21)),
                   'weights': ["uniform","distance"],
                   'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
        knn = KNeighborsClassifier()
        knn_randomcv = RandomizedSearchCV(estimator=knn, param_distributions=knn_grid, n_iter=100, cv = 5, scoring = "accuracy").fit(X_train,y_train)
        print("Best training score: {} with parameters: {}".format(knn_randomcv.best_score_, knn_randomcv.best_params_))
        print()
    
        knn = KNeighborsClassifier(**knn_randomcv.best_params_)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        
        
        accuracy_test = accuracy_score(y_test, y_pred)
        print("Test score: {}".format(accuracy_test))
        print()
        
        return knn_randomcv