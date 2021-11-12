from pycaret.classification import *

class ClassificationAutoML:
    def __init__(self):
        pass

    #setup classification automl
    def classificationAutoML(self,
                            data, 
                            targetName = '',
                            idColumnName = '',
                            trainSize = 0.7, 
                            random_seed = 1,
                            categoricalFeatures = [], 
                            numericFeatures = [],
                            ignoreFeatures = []):

        # check if user input target column name
        if targetName == '':
            y_actual_name = data.columns[-1]
        else:
            y_actual_name = targetName

        random_seed = 1

        s, column_and_datatype,  target_column_name = setup(data, 
                                                        target = y_actual_name, 
                                                        session_id = random_seed, 
                                                        train_size = trainSize,
                                                        categorical_features = categoricalFeatures,
                                                        numeric_features = numericFeatures,
                                                        ignore_features = ignoreFeatures,
                                                        silent = True)

        # convert label column to value "label"
        column_and_datatype[target_column_name] = "label"

        if idColumnName != '':
            column_and_datatype[idColumnName] = "ID Column"


        for index, val in column_and_datatype.iteritems():
            if "float" in str(val):
                column_and_datatype[index] = "Numerical"
            elif "object" in str(val):
                column_and_datatype[index] = "Categorical"
            elif "int" in str(val):
                column_and_datatype[index] = "Numericaa"        

        # convert to dataframe
        column_and_datatype_dataframe = column_and_datatype.to_frame(name = 'Data Type')
        column_and_datatype_dataframe = column_and_datatype_dataframe.reset_index()
        column_and_datatype_dataframe = column_and_datatype_dataframe.rename(columns={'index': 'Columns'})
        
        return column_and_datatype_dataframe



    def fitClassificationModel(self):
        # dateNow = datetime.now().strftime('%Y.%m.%d %H.%M.%S')
        # print("Today's Date: ", dateNow)
        best, results = compare_models()
        return best, results

    def save(self, best):
        save_model(best, 'classification_model')

    def tune(self, model):
        tuned_dt = tune_model(model)
        return tuned_dt

