import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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



def categoric_data(dataframe, cat_th = 10, car_th=20):
    """ 
    It serves to determine whether the variables in the dataset are categorical, numerical or cardinal variables.

    Parameters:
    ----------------
        dataframe: dataframe
                dataframe that wants to apply
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optional
                Class threshold for categorical but cardinal variables
                
    Returns:
    ---------------
    cat_cols: list
            Categorical variable list
    num_cols: list
            Numerical variable list
    cat_but_car: list
            Categorical but cardinal variable list
    num_but_cat: list
            Numerical but categorical variable list
            
    Notes:
    ---------------
    cat_cols + num_cols + cat_but_car = total variables
    num_but_cat variables are in cat_cols.
    """
    
    cat_cols = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["category", "bool", "object"]]
    num_but_cat = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["int64", "float64"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["category", "object"] and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variebles: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    
    return cat_cols, num_cols, cat_but_car


def evaluation(model, x_train, x_test, y_train, y_test):

    accuracy, precision, recall, f1 = {}, {}, {}, {}

    for key in model.keys():
        
        # Fit the classifier
        model[key].fit(x_train, y_train)
        
        # Make predictions
        predictions = model[key].predict(x_test)
        
        # Calculate metrics
        accuracy[key] = accuracy_score(y_test, predictions)
        precision[key] = precision_score(y_test, predictions)
        recall[key] = recall_score(y_test, predictions)
        f1[key] = f1_score(y_test, predictions)


    df_model = pd.DataFrame(index=model.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['F1 Score'] = f1.values()


    return df_model