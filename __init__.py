import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import streamlit as st
import classification as classify
import regression as reg
import cluster
import pandas as pd
import mail
import os

# sns.set_theme(style="whitegrid")
# ax2 = sns.barplot(x="", y="", data="")

path = [os.getcwd()+'\\Images\\classi', os.getcwd()+'\\Images\\reg', os.getcwd()+'\\Images\\classi']
for i in path:
    if not os.path.isdir(i):
        os.makedirs(i)


def plot_graph(model, how):
    try:
        if how == "":
            return classify.plot_model(model)
        else:
            return classify.plot_model(model, plot=how)
    except:
        return None

def upload():
    st.header(uploader.name, ' upload successfully')
    index = st.checkbox("Include first column as index")
    if index:
        df = pd.read_csv(uploader)
    else:
        df = pd.read_csv(uploader, index_col=False)
    st.write(df.head())
    if len(df.index) > 5:
        str1 = "There are " + str(len(df.index)) + " rows of data, only the first 5 is shown."
        st.caption(str1)
    # create_graph()


def run_classification_model(df, data_split, seed, label, id, column):
    class_model = classify.ClassificationAutoML()
    columns = []
    for i in column:
        if not column[i] and i != '':
            columns.append(i)
    st.markdown('<hr>', unsafe_allow_html=True)
    with st.spinner('The setup is loading...'):
        results = class_model.classificationAutoML(df, trainSize=data_split, random_seed=seed,
                                                                  targetName=label,
                                                                  idColumnName=id, ignoreFeatures=columns)
    with st.container():
        st.header("Data Type")
        st.write(results)
    data_type = st.selectbox("Is the data type correct?", ["Yes", "No"])
    proceed = st.checkbox("Proceed", key=1)
    proceed1 = False
    categorical, numerical = [], []
    if proceed and data_type is "No":
        type = {}
        for i in column:
            if column[i]:
                type[i] = st.selectbox(i, ['Categorical', 'Numerical'])
        for i in column:
            if i not in columns:
                if type[i] is 'Categorical':
                    categorical.append(i)
                elif type[i] is 'Numerical':
                    numerical.append(i)
        st.write(categorical)
        st.write(numerical)
        proceed1 = st.checkbox("Proceed", key=2)
        if proceed1:
            with st.spinner('The setup is loading...'):
                st.write(split_size)
                class_model.classificationAutoML(df, trainSize=split_size, random_seed=seed_number,
                                           categoricalFeatures=categorical,
                                           numericFeatures=numerical,
                                           targetName=label,
                                           idColumnName=id, ignoreFeatures=columns)
    elif (proceed and data_type is "Yes") or proceed1:
        st.markdown('<hr>', unsafe_allow_html=True)
        with st.spinner('The model is under training, it might take few minutes to execute'):
            best, results = class_model.fitClassificationModel()

        st.header("Comparison")
        st.write(results)
        results = results.data

        st.subheader("Accuracy graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='Accuracy', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\classi\\Accuracy.png')
        plt.clf()

        st.subheader("AUC graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='AUC', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\classi\\AUC.png')
        plt.clf()

        st.subheader("F1 graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='F1', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\classi\\F1.png')
        plt.clf()

        st.markdown('<hr>', unsafe_allow_html=True)
        st.header('The best model is ' + results.Model[0])
        st.subheader('Best parameter to use')
        st.code(best)
        st.subheader('Performance of ' + results.Model[0])
        plot_graph(best, '')
        plot_graph(best, 'confusion_matrix')
        plot_graph(best, 'class_report')

        tuning = st.checkbox("Perform Hyperparameter Tuning?")
        if tuning:
            with st.spinner('Tuning might take some time, please be patience'):
                best = class_model.tune(best)
            st.subheader('Parameter of tuned model')
            st.code(best)
        download = st.button('Download model')
        if download:
            class_model.save(best)
        st.write("check out this [link](https://colab.research.google.com/drive/1EJ_feSoTAnxcbM0wnfzlqGo5Vg6Vf7XI?usp=sharing) on how to implement the downloaded model")


def run_regression_model(df, data_split, seed, label, id, column):
    reg_model = reg.RegressionAutoML()
    columns = []
    for i in column:
        if not column[i] and i != '':
            columns.append(i)
    st.markdown('<hr>', unsafe_allow_html=True)
    with st.spinner('The setup is loading...'):
        results = reg_model.regressionAutoML(df, trainSize=data_split, random_seed=seed, targetName=label,
                                             idColumnName=id, ignoreFeatures=columns)
    with st.container():
        st.header("Data Type")
        st.write(results)
    data_type = st.selectbox("Is the data type correct?", ["Yes", "No"])
    proceed = st.checkbox("Proceed", key=1)
    proceed1 = False
    categorical, numerical = [], []
    if proceed and data_type is "No":
        type = {}
        for i in column:
            if column[i]:
                type[i] = st.selectbox(i, ['Categorical', 'Numerical'])
        for i in column:
            if i not in columns:
                if type[i] is 'Categorical':
                    categorical.append(i)
                elif type[i] is 'Numerical':
                    numerical.append(i)
        st.write(categorical)
        st.write(numerical)
        proceed1 = st.checkbox("Proceed", key=2)
        if proceed1:
            with st.spinner('The setup is loading...'):
                reg_model.regressionAutoML(df, trainSize=split_size, random_seed=seed_number,
                                           categoricalFeatures=categorical,
                                           numericFeatures=numerical,
                                           targetName=label,
                                           idColumnName=id, ignoreFeatures=columns)
    elif (proceed and data_type is "Yes") or proceed1:
        st.markdown('<hr>', unsafe_allow_html=True)
        with st.spinner('The model is under training, it might take few minutes to execute'):
            best, results = reg_model.fitRegressionModels()

        st.header("Comparison")
        st.write(results)
        results = results.data

        st.subheader("MAE graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='MAE', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\reg\\MAE.png')
        plt.clf()

        st.subheader("MSE graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='MSE', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\reg\\MSE.png')
        plt.clf()

        st.subheader("RMSE graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='RMSE', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\reg\\RMSE.png')
        plt.clf()

        st.subheader("R2 graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='R2', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\R2.png')
        plt.clf()

        with st.container():
            st.markdown('<hr>', unsafe_allow_html=True)
            st.header('The best model is ' + results.Model[0])
            st.subheader('Best parameter to use')
            st.code(best)
            st.subheader('Performance of '+ results.Model[0])
            plot_graph(best, 'vc')
            plot_graph(best, 'error')
            plot_graph(best, 'residuals')

        tuning = st.checkbox("Perform Hyperparameter Tuning?")
        if tuning:
            with st.spinner('Tuning might take some time, please be patience'):
                best = reg_model.tune(best)
            st.subheader('Parameter of tuned model')
            st.code(best)
        download = st.button('Download model')
        if download:
            reg_model.save(best)




def run_cluster_model(df, data_split, seed,  id, column):
    cluster_model = cluster.ClusterAutoML()
    columns = []
    for i in column:
        if not column[i] and i != '':
            columns.append(i)
    st.markdown('<hr>', unsafe_allow_html=True)
    with st.spinner('The setup is loading...'):
        results = cluster_model.clusterAutoML(df, trainSize=data_split, random_seed=seed,
                                             idColumnName=id, ignoreFeatures=columns)
    with st.container():
        st.header("Data Type")
        st.write(results)
    data_type = st.selectbox("Is the data type correct?", ["Yes", "No"])
    proceed = st.checkbox("Proceed", key=1)
    proceed1 = False
    categorical, numerical = [], []
    if proceed and data_type is "No":
        type = {}
        for i in column:
            if column[i]:
                type[i] = st.selectbox(i, ['Categorical', 'Numerical'])
        for i in column:
            if i not in columns:
                if type[i] is 'Categorical':
                    categorical.append(i)
                elif type[i] is 'Numerical':
                    numerical.append(i)
        st.write(categorical)
        st.write(numerical)
        proceed1 = st.checkbox("Proceed", key=2)
        if proceed1:
            with st.spinner('The setup is loading...'):
                cluster_model.clusterAutoML(df, trainSize=split_size, random_seed=seed_number,
                                           categoricalFeatures=categorical,
                                           numericFeatures=numerical,
                                           idColumnName=id, ignoreFeatures=columns)
    elif (proceed and data_type is "Yes") or proceed1:
        st.markdown('<hr>', unsafe_allow_html=True)
        with st.spinner('The model is under training, it might take few minutes to execute'):
            results, result, best = cluster_model.get_models()

        st.header("Comparison")
        st.write(results)

        st.subheader("Silhouette graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='Silhouette', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\cluster\\Silhouette.png')
        plt.clf()

        st.subheader("Calinski-Harabasz graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='Calinski-Harabasz', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\cluster\\Calinski-Harabasz.png')
        plt.clf()

        st.subheader("Davies-Bouldin graph")
        plt.figure(figsize=(6, 2))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=results.index, y='Davies-Bouldin', data=results)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.savefig(os.getcwd() + '\\Images\\cluster\\Davies-Bouldin.png')
        plt.clf()

        st.markdown('<hr>', unsafe_allow_html=True)
        st.header('The best model is '+results.model[0])
        st.subheader('Best parameter to use')
        st.code(best)
        st.subheader('Performance of '+ results.model[0])
        plot_graph(best, 'distance')
        plot_graph(best, 'elbow')
        plot_graph(best, 'silhouette')

        download = st.button('Download model')
        if download:
            cluster_model.save(best)


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


# Main page
st.title("MSG AutoML")
st.write("Developed by: Millenium Square Gang")

# chart = st.line_chart(last_rows)
# /Main page

def print_table(uploader, index=False):
    df = pd.read_csv(uploader, index_col=index)
    st.dataframe(df.head())

# Side bar
# progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
with st.sidebar.header("1. Upload your CSV data"):
    uploader = st.sidebar.file_uploader("Upload your input CSV file", help="Upload your dataset file here", type=['csv','xlsx'])
with st.sidebar.header('2. Choose model'):
    model_type = st.sidebar.selectbox("Please choose a model to predict", ["Classification", "Regression", "Clustering"])
with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
with st.sidebar.header("4. Email"):
    receivers = st.sidebar.text_area('User email(s)*', help='Separate multiple email by coma ( , )')
    cc = st.sidebar.text_area('CC email(s)', help='Separate multiple email by coma ( , )')
# /Sidebar

# Dataset
if uploader is not None and receivers != '':
    st.header(uploader.name+' preview')
    df = pd.read_csv(uploader, index_col=False)
    st.dataframe(df.head())
    if len(df.index) > 5:
        str1 = "There are " + str(len(df.index)-1) + " rows of data, only the first 5 is shown."
        st.caption(str1)
    label = ''
    if model_type is not "Clustering":
        st.subheader("Choose the Label column")
        label = st.selectbox('Label column', df.columns, index=len(df.columns) - 1)
    st.subheader("Choose the ID column")
    ids = df.columns.tolist()
    ids.insert(0, "--Not Selected--")
    id = st.selectbox('ID column', ids, index=0)
    if id == "--Not Selected--":
        id=''
    st.subheader("Choose feature")
    column = {}
    for i in df.columns:
        if not (i == id or i == label):
            column[i] = st.checkbox(i, value=True)

    run = st.checkbox("Click to Run")
    if run:
        path = os.getcwd() + "\\Images"
        if model_type is "Classification":
            run_classification_model(df, split_size/100.0, seed_number, label, id, column)
            path = path + '\\classi'
        elif model_type is "Regression":
            run_regression_model(df, split_size/100.0, seed_number, label, id, column)
            path = path + '\\reg'
        elif model_type is "Clustering":
            run_cluster_model(df, split_size/100.0, seed_number, id, column)
            path = path + '\\cluster'
        mail.sendReport(path, model_type, receivers, cc=cc)
else:
    st.info('Awaiting for dataset to be uploaded or user email')
