from pycaret.datasets import get_data
from pycaret.clustering import *

class ClusterAutoML:
    def __init__(self):
        pass
    
    def clusterAutoML(self,
                      dataset,
                      idColumnName='',
                      trainSize=0.7,
                      random_seed = 1,
                      categoricalFeatures = [], 
                      numericFeatures = [],
                      ignoreFeatures = [],
                      ):

        data = dataset.sample(frac=trainSize, random_state=random_seed).reset_index(drop=True)
        s, column_and_datatype,  target_column_name = setup(data,
                                                        normalize = True,
                                                        session_id = random_seed,
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
                column_and_datatype[index] = "Numeric"
            elif "object" in str(val):
                column_and_datatype[index] = "Categorical"
            elif "int" in str(val):
                column_and_datatype[index] = "Numeric"

        # convert to dataframe
        column_and_datatype_dataframe = column_and_datatype.to_frame(name = 'Data Type')
        column_and_datatype_dataframe = column_and_datatype_dataframe.reset_index()
        column_and_datatype_dataframe = column_and_datatype_dataframe.rename(columns={'index': 'Columns'})
        
        return column_and_datatype_dataframe 


    def get_models(self):
        model = []

        kmeans, kmeans_result = create_model('kmeans')
        kmode, kmode_result = create_model('kmodes')
        ap, ap_result = create_model('ap')
        meanshift, meanshift_result = create_model('meanshift')
        sc, sc_result = create_model('sc')
        hclust, hclust_result = create_model('hclust')
        dbscan, dbscan_result = create_model('dbscan')
        optics, optics_result = create_model('optics')
        birch, birch_result = create_model('birch')

        models=[kmeans,kmode,ap,meanshift,sc,hclust,dbscan,optics,birch]

        df_full = pd.concat([kmeans_result,kmode_result,ap_result,meanshift_result,sc_result,hclust_result,dbscan_result,optics_result,birch_result])
        df_full['model'] = ['kmeans' , 'kmode', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch']

        df_new = df_full.reset_index(drop=True)
        cols = list(df_new.columns)
        cols = cols[-1:] + cols[:-1]
        df_new = df_new[cols]
        df_new.sort_values(['Silhouette'], ascending=[False])
        # print(df_new)

        best = df_new['Silhouette'].idxmax()
        # tuned_best = tune_model(model=models[best],supervised_target='class')

        result = assign_model(models[best])

        return df_new, result, models[best]

    def save(self, best):
        save_model(best, 'clustering_model')

    







