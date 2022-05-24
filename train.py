import resssssssssssssssssss
import math
import json
import numpy as np
import pandas as pd
import nbformat as nbf
from io import StringIO
from utils_framework import preprossessteps
from utils_text_train import preprossessteps_train
from utils_text_threshold import threshold_functions
from utils_text_predict import preprossessteps_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from utils_sanitize import sanitize_preprossessteps
from sklearn.compose import make_column_transformer

def find_threshold(values, k):
    vals = pd.Series(values)
    mean = vals.mean()
    std = vals.std()
    if pd.isna(std):
        std = 0
    return math.ceil(mean + k*std)

def iqr_bounds(scores, k=3):
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return upper_bound

def join_lists(lists):
    output = []
    for lst in lists:
        output = output + list(lst)
    return set(output)

def join_lists_text():
    outtext = '''def join_lists(lists):
    output = []
    for lst in lists:
        output = output + list(lst)
    return set(output)'''
    return outtext

def read_json_file(path):
    with open(path, 'r') as f:
            metadata = json.load(f)
    return metadata

def save_json_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_preprocess_metadata(result):
    result_keys = result.keys()
    ids = [x.split('_')[-1] for x in result_keys if 'function_type_' in x]
    col_name =[]
    func_name =[]
    for col_id in ids:
        col_name.append(result.getlist('column_name_' + col_id))
        func_type = result['function_type_' + col_id]
        func_name.append([func_type])
        if 'option_' + col_id in result:
            func_name[-1].append(result.getlist('option_' + col_id))

    preprocess_steps = list(zip(col_name, func_name))
    return preprocess_steps

def get_processed_data(df, preprocess_steps, column_list=None):
    ppdata = preprossessteps(df)
    for column, func_option in preprocess_steps:
        if func_option[0] == 'label_encoding':
            ppdata.lable_encoding(column=column[0])

        elif func_option[0] == 'fill_na':
            ppdata.fill_na(column=column, value=func_option[1][0])

        elif func_option[0] == 'drop_na':
            ppdata.drop_na(column=column)

        elif func_option[0] == 'drop_duplicate':
            ppdata.drop_duplicate(column=column)

        elif func_option[0] == 'one_hot_encode':
            ppdata.one_hot_encode(column=column)

        elif func_option[0] == 'tfidf' and (column_list==None or column+'_tfidf' not in column_list):
            ppdata.tfidf_vectorization(column=column[0])

        elif func_option[0] == 'count_vect' and (column_list==None or column+'_count_vect' not in column_list):
            ppdata.count_vectorization(column=column[0])

        elif func_option[0] == 'min_max':
            ppdata.MinMaxScaler(column=column)

        elif func_option[0] == 'split_to_list':
            ppdata.split_to_list(column=column[0], split_pattern=func_option[1][0])

        elif func_option[0] == 'filter':
            ppdata.filter(column=column[0], values=func_option[1])
    df = ppdata.get_df()
    return df

def get_import_text(preprocess_steps):
    outtext = ""
    functions = set([x[1][0] for x in preprocess_steps])
    for func in functions:
        if func == 'label_encoding':
            outtext += "from sklearn.preprocessing import LabelEncoder"
        elif func == 'one_hot_encode':
            outtext += "from sklearn.preprocessing import OneHotEncoder"
        elif func == 'tfidf':
            outtext += "from sklearn.feature_extraction.text import TfidfVectorizer"
            outtext += "\n"
            outtext += "from sklearn.compose import make_column_transformer"
        elif func == 'count_vect':
            outtext += "from sklearn.feature_extraction.text import CountVectorizer"
            outtext += "\n"
            outtext += "from sklearn.compose import make_column_transformer"
        elif func == 'min_max':
            outtext += "from sklearn.preprocessing import MinMaxScaler"
        elif func == 'fill_na':
            outtext += "from sklearn.impute import SimpleImputer"
        outtext += "\n"
    outtext += "\n"
    return outtext

def get_preprocess_text(preprocess_steps, step, column_list=None):
    if step == 'train':
        ppdata = preprossessteps_train()
    elif step =='predict':
        ppdata = preprossessteps_predict()
    outtext = ""
    for column, func_option in preprocess_steps:
        if func_option[0] == 'label_encoding':
            outtext += ppdata.lable_encoding(column=column[0])

        elif func_option[0] == 'fill_na':
            outtext += ppdata.fill_na(column=column, value=func_option[1][0])

        elif func_option[0] == 'drop_na':
            outtext += ppdata.drop_na(column=column)

        elif func_option[0] == 'drop_duplicate':
            outtext += ppdata.drop_duplicate(column=column)

        elif func_option[0] == 'one_hot_encode':
            outtext += ppdata.one_hot_encode(column=column)

        elif func_option[0] == 'tfidf' and (column_list==None or column+'_tfidf' not in column_list):
            outtext += ppdata.tfidf_vectorization(column=column[0])

        elif func_option[0] == 'count_vect' and (column_list==None or column+'_count_vect' not in column_list):
            outtext += ppdata.count_vectorization(column=column[0])

        elif func_option[0] == 'min_max':
            outtext += ppdata.MinMaxScaler(column=column)
        
        elif func_option[0] == 'split_to_list':
            outtext += ppdata.split_to_list(column=column[0], split_pattern=func_option[1][0])

        elif func_option[0] == 'filter':
            outtext += ppdata.filter(column=column[0], values=func_option[1])
        outtext += "\n    "
    return outtext

def import_library(model):
    outtext = ""
    if model == 'Logistic Regression':
        outtext += "from sklearn.linear_model import LogisticRegression"
    elif model == 'Random Forest':
        outtext += "from sklearn.ensemble import RandomForestClassifier"
    elif model == 'knn':
        outtext += "from sklearn.neighbors import KNeighborsClassifier"
    elif model == 'Decision Tree':
        outtext += "from sklearn.tree import DecisionTreeClassifier"
    elif model == 'Support Vector Machine':
        outtext += "from sklearn.svm import SVC"
    elif model == 'Linear Regression':
        outtext += "from sklearn.linear_model import LinearRegression"
    elif model == 'Polynomial Regression':
        outtext += "from sklearn.preprocessing import PolynomialFeatures"
        outtext += "\nfrom sklearn.linear_model import LinearRegression"
    elif model == 'Random Forest Regression':
        outtext += "from sklearn.ensemble import RandomForestRegressor"
    return outtext

def get_model_train_text(model, options={}):
    outtext = ""
    if model == 'Logistic Regression':
        if options == {}:
            outtext += "LogisticRegression(random_state=0)"
        else:
            outtext += "LogisticRegression(**" + str(options) + ")"
    elif model == 'Random Forest':
        if options == {}:
            outtext += "RandomForestClassifier(random_state = 0)"
        else:
            outtext += "RandomForestClassifier(**" + str(options) + ")"
    elif model == 'knn':
        if options == {}:
            outtext += "KNeighborsClassifier()"
        else:
            outtext += "KNeighborsClassifier(**" + str(options) + ")"
    elif model == 'Decision Tree':
        if options == {}:
            outtext += "DecisionTreeClassifier(random_state=0)"
        else:
            outtext += "DecisionTreeClassifier(**" + str(options) + ")"
    elif model == 'Support Vector Machine':
        if options == {}:
            outtext += "SVC(random_state=0)"
        else:
            outtext += "SVC(**" + str(options) + ")"
    elif model == 'Linear Regression':
        if options == {}:
            outtext += "LinearRegression()"
        else:
            outtext += "LinearRegression(**" + str(options) + ")"
    elif model == 'Polynomial Regression':
        if options == {}:
            outtext += "make_pipeline(PolynomialFeatures(), LinearRegression())"
        else:
            outtext += "make_pipeline(PolynomialFeatures(degree=" + str(options['degree']) + "), LinearRegression(fit_intercept=" + str(options['fit_intercept']) + "))"
    elif model == 'Random Forest Regression':
        if options == {}:
            outtext += "RandomForestRegressor(random_state=0)"
        else:
            outtext += "RandomForestRegressor(**" + str(options) + ")"
    return outtext

def get_model_train(model, options={}):
    if model == 'Logistic Regression':
        if options == {}:
            return LogisticRegression(random_state=0)
        else:
            return LogisticRegression(**options)
    elif model == 'Random Forest':
        if options == {}:
            return RandomForestClassifier(random_state = 0)
        else:
            return RandomForestClassifier(**options)
    elif model == 'knn':
        if options == {}:
            return KNeighborsClassifier()
        else:
            return KNeighborsClassifier(**options)
    elif model == 'Decision Tree':
        if options == {}:
            return DecisionTreeClassifier(random_state=0)
        else:
            return DecisionTreeClassifier(**options)
    elif model == 'Support Vector Machine':
        if options == {}:
            return SVC(random_state=0)
        else:
            return SVC(**options)
    elif model == 'Linear Regression':
        if options == {}:
            return LinearRegression()
        else:
            return LinearRegression(**options)
    elif model == 'Polynomial Regression':
        if options == {}:
            return make_pipeline(PolynomialFeatures(),LinearRegression())
        else:
            return make_pipeline(PolynomialFeatures(degree=options['degree']),LinearRegression(fit_intercept=options['fit_intercept']))
    elif model == 'Random Forest Regression':
        if options == {}:
            return RandomForestRegressor(random_state=0)
        else:
            return RandomForestRegressor(**options)

def make_column_transform_pipeline(df, metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list'].copy()
    column_transform_list = []
    metadata['hide_columns'] = []
    is_pca = False
    for column, func_option in preprocess_steps:
        if func_option[0] == 'tfidf':
            metadata['hide_columns'].append(column[0] + '_tfidf')
        elif func_option[0] == 'count_vect':
            metadata['hide_columns'].append(column[0] + '_count_vect')
        if (column[0] + '_tfidf') in column_list or (column[0] + '_count_vect') in column_list :
            if func_option[0] == 'tfidf':
                column_transform_list.append((TfidfVectorizer(analyzer='word', stop_words= 'english'), column[0]))
                column_list.remove(column[0] + '_tfidf')
                column_list.append(column[0])
            elif func_option[0] == 'count_vect':
                column_transform_list.append((CountVectorizer(analyzer='word', stop_words= 'english'), column[0]))
                column_list.remove(column[0] + '_count_vect')
                column_list.append(column[0])
    if column_transform_list:
        column_trans = make_column_transformer(*column_transform_list, remainder='passthrough')
        x_train = column_trans.fit_transform(df[column_list])
        is_pca = True
    else:
        x_train = df[column_list]
    return x_train, is_pca

def get_column_transform_pipeline_text(metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list'].copy()
    column_transform_list = []
    outtext_fit = ""
    outtext_predict = ""
    outtext_transform_train = ""
    outtext_transform_predict = ""
    is_pca = False
    for column, func_option in preprocess_steps:
        if (column[0] + '_tfidf') in column_list or (column[0] + '_count_vect') in column_list :
            if func_option[0] == 'tfidf':
                column_transform_list.append((TfidfVectorizer(analyzer='word',stop_words= 'english'), column[0]))
                column_list.remove(column[0] + '_tfidf')
                column_list.append(column[0])
            elif func_option[0] == 'count_vect':
                column_transform_list.append((CountVectorizer(analyzer='word',stop_words= 'english'), column[0]))
                column_list.remove(column[0] + '_count_vect')
                column_list.append(column[0])
    if column_transform_list:
        outtext_fit += "column_trans = make_column_transformer(" + str(column_transform_list)[1:-1] + ", remainder='passthrough')"
        outtext_fit += "\n    "
        outtext_fit += "column_trans.fit(df[" + str(column_list) + "])"
        outtext_fit += "\n    "
        outtext_fit += "with open('column_trannsformer.pkl', 'wb') as f:"
        outtext_fit += "\n        "
        outtext_fit += "pickle.dump(column_trans, f)"
        outtext_fit += "\n    "
        outtext_predict = "with open('column_trannsformer.pkl', 'rb') as f:"
        outtext_predict += "\n        "
        outtext_predict += "column_trans = pickle.load(f)"
        outtext_predict += "\n    "
        outtext_transform_train += "column_trans.transform(df[" + str(column_list) + "])"
        outtext_transform_predict += "column_trans.transform(df_test[" + str(column_list) + "])"
        is_pca = True
    else:
        outtext_transform_train += "df[" + str(column_list) + "]"
        outtext_transform_predict += "df_test[" + str(column_list) + "]"
    return outtext_fit, outtext_predict, outtext_transform_train, outtext_transform_predict, is_pca

def prepare_py_files_stats(metadata):
    preprocess_steps = metadata['preprocess_steps']
    import_text = get_import_text(preprocess_steps)
    preprocess_text_train = get_preprocess_text(preprocess_steps, step="train")
    preprocess_text_predict = get_preprocess_text(preprocess_steps, step="predict")

    thf = threshold_functions()
    if metadata['threshold_func'] == 'mean_std':
        thf_text = thf.standard_dev()
    elif metadata['threshold_func'] == 'iqr_bound':
        thf_text = thf.iqr_bounds()

    k_value = '2.5'
    if 'k_value' in metadata:
        k_value = str(metadata['k_value'])

    group_column = metadata['group_column']
    date_column = metadata['date_column']
    count_column_list = metadata['count_column_list']

    count_lists = []
    new_column_names = []
    for count_column in count_column_list:
        new_column_name = re.sub(r'[^a-zA-Z0-9_]', '_', count_column)
        if new_column_name[0].isalpha() or new_column_name[0] == '_':
            pass
        else:
            new_column_name = '_' + new_column_name
        new_column_names.append(new_column_name)
        temp = new_column_name + '_hist = (' + '\'' + count_column + '\'' + ', list)'
        count_lists.append(temp)

    count_group_lists = ', '.join(count_lists)

    group_lists = count_group_lists
    agg_dict = {}

    predict_text = ""
    if 'save_hist' in metadata:
        import_text += join_lists_text()
        import_text += '\n'
        hist_column_list = metadata['hist_column']
        hist_lists = []

        for h_column in hist_column_list:
            new_column_name = re.sub(r'[^a-zA-Z0-9_]', '_', h_column)
            if new_column_name[0].isalpha() or new_column_name[0] == '_':
                pass
            else:
                new_column_name = '_' + new_column_name
            temp = new_column_name + '_hist = (' + '\'' + h_column + '\'' + ', join_lists)'
            hist_lists.append(temp)
            agg_dict[h_column] = 'join_lists'

            predict_text += "\n    "
            predict_text += "df_output['" + new_column_name + "_hist'] = df_output['" + new_column_name + "_hist'].fillna('')"
            predict_text += "\n    "
            predict_text += "df_output['new_" + h_column + "'] = df_output[['" + h_column + "', '" + new_column_name + "_hist']]"
            predict_text +=  ".apply(lambda x:set(x['" + h_column + "']) - set(x['" + new_column_name + "_hist']), axis=1)"
            predict_text += "\n    "
            predict_text += "df_output['" + h_column + "_anomaly'] = [-1 if x else 1 for x in df_output['new_" + h_column + "']]"
            predict_text += "\n    "
            predict_text += "df_output['new_" + h_column + "'] = df_output['new_" + h_column + "'].replace(set(), '')"

        hist_group_lists = ', '.join(hist_lists)
        group_lists = count_group_lists + ', ' + hist_group_lists

    column_type = metadata['column_type']
    extra_step = ""
    extra_step_predict = ""
    for count_column in count_column_list:
        if column_type[count_column] == 'int64' or column_type[count_column] == 'float64':
            agg_dict[count_column] = 'sum'
        else:
            agg_dict[count_column] = 'nunique'
    extra_step += "df = df.groupby(" + str(group_column + [date_column]) + ").agg(" + str(agg_dict).replace("'join_lists'", "join_lists") + ").reset_index()"
    extra_step_predict += "df_test = df_test.groupby(" + str(group_column + [date_column]) + ").agg(" + str(agg_dict).replace("'join_lists'", "join_lists") + ").reset_index()"

    if 'num_days' in metadata:
            num_days_group = "num_days_data = ('" + str(date_column) + "', 'nunique')"
            group_lists = num_days_group + ', ' + group_lists

    new_group_text = ""
    if 'new_group' in metadata:
        new_group_text += "\n    "
        new_group_text += "df_output['"+new_column_names[0]+"_hist'] = df_output['"+new_column_names[0]+"_hist'].fillna('')"
        new_group_text += "\n    "
        new_group_text += "df_output['new_group'] = df_output['"+new_column_names[0]+"_hist'].apply(lambda x: '' if x else 'New Group')"

    plt_thr_text_train = ""
    plt_thr_text_predict = ""
    if 'plt_threshold' in metadata:
        threshold_col = metadata['platform_count']
        if 'platform_column' in metadata:
            default_threshold_group = metadata['platform_column']
            plt_dict = {col+'_threshold' : 'mean' for col in threshold_col}
            plt_rename = {col+'_threshold': col+'_platform_threshold' for col in count_column_list}
            plt_thr_text_train += "\n    "
            plt_thr_text_train += "df_plt_threshold = df_group.groupby(" + str(default_threshold_group) + ").agg(" + str(plt_dict) + ").reset_index()"
            plt_thr_text_train += "\n    "
            plt_thr_text_train += "df_plt_threshold = df_plt_threshold.rename(columns = " + str(plt_rename) + ")"
            plt_thr_text_train += "\n    "
            plt_thr_text_train += "data_to_save['df_plt_threshold'] = df_plt_threshold"
            plt_thr_text_train += "\n    "

            plt_thr_text_predict += "\n    "
            plt_thr_text_predict += "df_plt_threshold = saved_data['df_plt_threshold']"
            plt_thr_text_predict += "\n    "
            plt_thr_text_predict += "df_output = df_output.merge(df_plt_threshold, on=" + str(default_threshold_group) + ", how='left')"
            plt_thr_text_predict += "\n    "
            plt_thr_text_predict += "for column in " + str(threshold_col) + ":"
            plt_thr_text_predict += "\n        "
            plt_thr_text_predict += "df_output[column + '_platform_anomaly'] = df_output[column] > df_output[column + '_platform_threshold']"
            plt_thr_text_predict += "\n        "
            plt_thr_text_predict += "df_output[column + '_platform_anomaly'] = df_output[column + '_platform_anomaly'].map({True:-1, False:1})"
            plt_thr_text_predict += "\n    "
        else:
            threshold_col_thr = [col+'_threshold' for col in threshold_col]
            plt_thr_text_train += "\n    "
            plt_thr_text_train += "plt_threshold = df_group[" + str(threshold_col_thr) + "].mean()"
            plt_thr_text_train += "\n    "
            plt_thr_text_train += "data_to_save['plt_threshold'] = plt_threshold"
            plt_thr_text_train += "\n    "

            plt_thr_text_predict += "\n    "
            plt_thr_text_predict += "plt_threshold = saved_data['plt_threshold']"
            plt_thr_text_predict += "\n    "
            plt_thr_text_predict += "for column in " + str(threshold_col) + ":"
            plt_thr_text_predict += "\n        "
            plt_thr_text_predict += "df_output[column + '_platform_threshold'] = plt_threshold.loc[column+'_threshold']"
            plt_thr_text_predict += "\n        "
            plt_thr_text_predict += "df_output[column + '_platform_anomaly'] = df_output[column] > df_output[column + '_platform_threshold']"
            plt_thr_text_predict += "\n        "
            plt_thr_text_predict += "df_output[column + '_platform_anomaly'] = df_output[column + '_platform_anomaly'].map({True:-1, False:1})"
            plt_thr_text_predict += "\n    "


    with open('py_ipynb_templates/threshold_template_train.py', 'r') as f:
        train = f.read()

    with open('py_ipynb_templates/threshold_template_prediction.py', 'r') as f:
        prediction = f.read()
    
    group_column_text = '\'' + '\', \''.join(group_column) + '\''
    count_column_text = '\'' + '\', \''.join(count_column_list) + '\''

    train = train.replace('k_value_factor', k_value)
    train = train.replace('#extra_imports', import_text).replace('#preprossessteps_train', preprocess_text_train)
    train = train.replace('#extra_step', extra_step)
    train = train.replace('#threshold_function', thf_text)
    train = train.replace('column_to_group_list', group_column_text).replace('count_column_list', count_column_text).replace('count_group_lists', group_lists)
    train = train.replace('#platform_threshold_text', plt_thr_text_train)
    
    prediction = prediction.replace('#preprossessteps_predict', preprocess_text_predict)
    prediction = prediction.replace('#extra_step', extra_step_predict)
    prediction = prediction.replace('column_to_group_list', group_column_text).replace('count_column_list', count_column_text)
    prediction = prediction.replace('#hist_data_code', predict_text)
    prediction = prediction.replace('#new_group_identify', new_group_text)
    prediction = prediction.replace('#platform_threshold_text', plt_thr_text_predict)

    return train, prediction

def prepare_py_files_isoforest(metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list']
    import_text = get_import_text(preprocess_steps)
    preprocess_text_train = get_preprocess_text(preprocess_steps, step="train", column_list=column_list)
    preprocess_text_predict = get_preprocess_text(preprocess_steps, step="predict", column_list=column_list)

    extra_preprocess_train, extra_preprocess_predict, x_train_text, x_test_text, is_pca = get_column_transform_pipeline_text(metadata)
    preprocess_text_train += extra_preprocess_train
    preprocess_text_predict += extra_preprocess_predict

    contamination = '0.05'
    if 'contamination' in metadata:
        contamination = str(metadata['contamination'])

    with open('py_ipynb_templates/isolation_template_train.py', 'r') as f:
        train = f.read()

    with open('py_ipynb_templates/isolation_template_prediction.py', 'r') as f:
        prediction = f.read()

    train = train.replace('contamination_factor', contamination)
    train = train.replace('#extra_imports', import_text).replace('#preprossessteps_train', preprocess_text_train)
    train = train.replace("'x_train_text'", x_train_text)
    prediction = prediction.replace('#preprossessteps_predict', preprocess_text_predict)
    prediction = prediction.replace("'x_test_text'", x_test_text)

    return train, prediction

def prepare_py_files_kMeans(metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list']
    import_text = get_import_text(preprocess_steps)
    preprocess_text_train = get_preprocess_text(preprocess_steps, step="train", column_list=column_list)
    preprocess_text_predict = get_preprocess_text(preprocess_steps, step="predict", column_list=column_list)

    extra_preprocess_train, extra_preprocess_predict, x_train_text, x_test_text, is_pca = get_column_transform_pipeline_text(metadata)
    preprocess_text_train += extra_preprocess_train
    preprocess_text_predict += extra_preprocess_predict

    clusters = '3'
    if 'clusters' in metadata:
        clusters = str(metadata['clusters'])

    with open('py_ipynb_templates/KMeans_train.py', 'r') as f:
        train = f.read()

    with open('py_ipynb_templates/KMeans_prediction.py', 'r') as f:
        prediction = f.read()

    train = train.replace('no_of_clusters', clusters)
    train = train.replace('#extra_imports', import_text).replace('#preprossessteps_train', preprocess_text_train)
    train = train.replace("'x_train_text'", x_train_text)
    prediction = prediction.replace('#preprossessteps_predict', preprocess_text_predict)
    prediction = prediction.replace("'x_test_text'", x_test_text)

    with open('py_ipynb_templates/kmeans_notebook.ipynb', 'r') as f:
        ipynb = f.read()
    nb = nbf.v4.reads(ipynb)

    ec_min = '1'
    ec_max = '10'
    if 'ec_min' in metadata:
        ec_min = str(metadata['ec_min'])
        ec_max = str(metadata['ec_max'])

    graph_text_train = ""
    graph_text_predict = ""
    if len(column_list) == 1 and not is_pca:
        graph_text_train += "fig = px.histogram(data, x='" + column_list[0] + "', color=df['Cluster'])"
        graph_text_predict += "fig3 = px.histogram(test_data, x='" + column_list[0] + "', color=df_test['Cluster'])"
    elif len(column_list) == 2 and not is_pca:
        graph_text_train += "fig = px.scatter(data, x='" + column_list[0] + "', y='" + column_list[1] + "', color=df['Cluster'])"
        graph_text_predict += "fig3 = px.scatter(test_data, x='" + column_list[0] + "', y='" + column_list[1] + "', color=df_test['Cluster'])"
    elif is_pca:
        graph_text_train += "pca = PCA(n_components=2).fit_transform(data.toarray())\n"
        graph_text_train += "fig = px.scatter(x=pca[:,0], y=pca[:,1], color=df['Cluster'])"
        graph_text_predict += "pca_test = PCA(n_components=2).fit_transform(test_data.toarray())\n"
        graph_text_predict += "fig3 = px.scatter(x=pca_test[:,0], y=pca_test[:,1], color=df_test['Cluster'])"
    else:
        graph_text_train += "pca = PCA(n_components=2).fit_transform(data)\n"
        graph_text_train += "fig = px.scatter(x=pca[:,0], y=pca[:,1], color=df['Cluster'])"
        graph_text_predict += "pca_test = PCA(n_components=2).fit_transform(test_data)\n"
        graph_text_predict += "fig3 = px.scatter(x=pca_test[:,0], y=pca_test[:,1], color=df_test['Cluster'])"

    preprocess_ipy_train = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_train.split('\n')]).strip()
    preprocess_ipy_predict = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_predict.split('\n')]).strip()
    nb['cells'][1]['source'] = nb['cells'][1]['source'].replace('#extra_imports', import_text).strip()
    nb['cells'][8]['source'] = nb['cells'][8]['source'].replace('#preprossessteps_train', preprocess_ipy_train)
    nb['cells'][10]['source'] = nb['cells'][10]['source'].replace("'x_train_text'", x_train_text)
    nb['cells'][12]['source'] = nb['cells'][12]['source'].replace('ec_min', ec_min).replace('ec_max', ec_max)
    nb['cells'][14]['source'] = nb['cells'][14]['source'].replace('no_of_clusters', clusters)
    nb['cells'][18]['source'] = nb['cells'][18]['source'].replace('#graph1', graph_text_train)
    nb['cells'][22]['source'] = nb['cells'][22]['source'].replace('#preprossessteps_predict', preprocess_ipy_predict)
    nb['cells'][23]['source'] = nb['cells'][23]['source'].replace("'x_test_text'", x_test_text)
    nb['cells'][27]['source'] = nb['cells'][27]['source'].replace('#graph3', graph_text_predict)

    f = StringIO()
    nbf.write(nb, f)
    f.seek(0)
    ipynb_text = f.read()

    return train, prediction, ipynb_text

def prepare_py_files_classification(metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list']
    import_text = get_import_text(preprocess_steps)
    preprocess_text_train = get_preprocess_text(preprocess_steps, step="train", column_list=column_list)
    preprocess_text_predict = get_preprocess_text(preprocess_steps, step="predict", column_list=column_list)

    extra_preprocess_train, extra_preprocess_predict, x_train_text, x_test_text, is_pca = get_column_transform_pipeline_text(metadata)
    preprocess_text_train += extra_preprocess_train
    preprocess_text_predict += extra_preprocess_predict

    model = metadata['model']
    import_text = import_library(model) + '\n' + import_text

    model_text = get_model_train_text(model, metadata['model_options'])

    predict_column = metadata['predict_column']
    with open('py_ipynb_templates/classification_template_train.py', 'r') as f:
        train = f.read()

    with open('py_ipynb_templates/classification_template_prediction.py', 'r') as f:
        prediction = f.read()

    train = train.replace('#extra_imports', import_text).replace('#preprossessteps_train', preprocess_text_train)
    train = train.replace("'x_train_text'", x_train_text)
    train = train.replace("predict_column", predict_column)
    train = train.replace("'classification_model'", model_text)
    prediction = prediction.replace('#preprossessteps_predict', preprocess_text_predict)
    prediction = prediction.replace("'x_test_text'", x_test_text)

    with open('py_ipynb_templates/cf_notebook.ipynb', 'r') as f:
        ipynb = f.read()
    nb = nbf.v4.reads(ipynb)

    graph_text_train = ""
    graph_text_predict = ""
    if len(column_list) == 1 and not is_pca:
        graph_text_train += "fig = px.histogram(data, x='" + column_list[0] + "', color=df['prediction'])"
        graph_text_predict += "fig3 = px.histogram(test_data, x='" + column_list[0] + "', color=df_test['prediction'])"
    elif len(column_list) == 2 and not is_pca:
        graph_text_train += "fig = px.scatter(data, x='" + column_list[0] + "', y='" + column_list[1] + "', color=df['prediction'])"
        graph_text_predict += "fig3 = px.scatter(test_data, x='" + column_list[0] + "', y='" + column_list[1] + "', color=df_test['prediction'])"
    else:
        graph_text_train += "pca = PCA(n_components=2).fit_transform(x_train)\n"
        graph_text_train += "fig = px.scatter(x=pca[:,0], y=pca[:,1], color=df['prediction'])"
        graph_text_predict += "pca_test = PCA(n_components=2).fit_transform(x_test)\n"
        graph_text_predict += "fig3 = px.scatter(x=pca_test[:,0], y=pca_test[:,1], color=df_test['prediction'])"

    preprocess_ipy_train = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_train.split('\n')]).strip()
    preprocess_ipy_predict = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_predict.split('\n')]).strip()
    nb['cells'][1]['source'] = nb['cells'][1]['source'].replace('#extra_imports', import_text).strip()
    nb['cells'][8]['source'] = nb['cells'][8]['source'].replace('#preprossessteps_train', preprocess_ipy_train)
    nb['cells'][10]['source'] = nb['cells'][10]['source'].replace("'x_train_text'", x_train_text)
    nb['cells'][10]['source'] = nb['cells'][10]['source'].replace("predict_column", predict_column)
    nb['cells'][12]['source'] = nb['cells'][12]['source'].replace("'classification_model'", model_text)
    nb['cells'][16]['source'] = nb['cells'][16]['source'].replace('#graph1', graph_text_train)
    nb['cells'][21]['source'] = nb['cells'][21]['source'].replace('#preprossessteps_predict', preprocess_ipy_predict)
    nb['cells'][22]['source'] = nb['cells'][22]['source'].replace("'x_test_text'", x_test_text)
    nb['cells'][22]['source'] = nb['cells'][22]['source'].replace("predict_column", predict_column)
    nb['cells'][26]['source'] = nb['cells'][26]['source'].replace('#graph3', graph_text_predict)

    f = StringIO()
    nbf.write(nb, f)
    f.seek(0)
    ipynb_text = f.read()

    return train, prediction, ipynb_text

def prepare_py_files_regression(metadata):
    preprocess_steps = metadata['preprocess_steps']
    column_list = metadata['column_list']
    import_text = get_import_text(preprocess_steps)
    preprocess_text_train = get_preprocess_text(preprocess_steps, step="train", column_list=column_list)
    preprocess_text_predict = get_preprocess_text(preprocess_steps, step="predict", column_list=column_list)

    extra_preprocess_train, extra_preprocess_predict, x_train_text, x_test_text, is_pca = get_column_transform_pipeline_text(metadata)
    preprocess_text_train += extra_preprocess_train
    preprocess_text_predict += extra_preprocess_predict

    model = metadata['model']
    import_text = import_library(model) + '\n' + import_text

    model_text = get_model_train_text(model, metadata['model_options'])

    predict_column = metadata['predict_column']
    with open('py_ipynb_templates/regression_template_train.py', 'r') as f:
        train = f.read()

    with open('py_ipynb_templates/regression_template_prediction.py', 'r') as f:
        prediction = f.read()

    train = train.replace('#extra_imports', import_text).replace('#preprossessteps_train', preprocess_text_train)
    train = train.replace("'x_train_text'", x_train_text)
    train = train.replace("predict_column", predict_column)
    train = train.replace("'regression_model'", model_text)
    prediction = prediction.replace('#preprossessteps_predict', preprocess_text_predict)
    prediction = prediction.replace("'x_test_text'", x_test_text)

    with open('py_ipynb_templates/rg_notebook.ipynb', 'r') as f:
        ipynb = f.read()
    nb = nbf.v4.reads(ipynb)

    preprocess_ipy_train = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_train.split('\n')]).strip()
    preprocess_ipy_predict = '\n'.join([line [4:] if line.startswith(' ') else line for line in preprocess_text_predict.split('\n')]).strip()
    nb['cells'][1]['source'] = nb['cells'][1]['source'].replace('#extra_imports', import_text).strip()
    nb['cells'][9]['source'] = nb['cells'][9]['source'].replace('#preprossessteps_train', preprocess_ipy_train)
    nb['cells'][11]['source'] = nb['cells'][11]['source'].replace("'x_train_text'", x_train_text)
    nb['cells'][11]['source'] = nb['cells'][11]['source'].replace("predict_column", predict_column)
    nb['cells'][13]['source'] = nb['cells'][13]['source'].replace("'regression_model'", model_text)
    nb['cells'][22]['source'] = nb['cells'][22]['source'].replace('#preprossessteps_predict', preprocess_ipy_predict)
    nb['cells'][23]['source'] = nb['cells'][23]['source'].replace("'x_test_text'", x_test_text)
    nb['cells'][23]['source'] = nb['cells'][23]['source'].replace("predict_column", predict_column)

    f = StringIO()
    nbf.write(nb, f)
    f.seek(0)
    ipynb_text = f.read()

    return train, prediction, ipynb_text

def get_hyperparameters(model, result, labels=None):
    hp = {}
    if model=='Random Forest':
        hp['n_estimators'] = int(result['n_estimators'])
        hp['criterion'] = result['criterion']
        hp['max_depth'] = None if int(result['max_depth']) == 0 else int(result['max_depth'])
        hp['min_samples_split'] = int(result['min_samples_split'])
        hp['min_samples_leaf'] = int(result['min_samples_leaf'])
        hp['max_features'] = result['max_features']
        hp['bootstrap'] = True if result['bootstrap'] == 'True' else False
        hp['random_state'] = int(result['random_state'])
        if str(result['class_weight']).lower().strip() == 'none':
            hp['class_weight'] = None
        elif result['class_weight'] == '':
            hp['class_weight'] = {label : float(result['class_weight_' + str(label)]) for label in labels}
        else:
            hp['class_weight'] = result['class_weight']
    elif model=='Logistic Regression':
        hp['penalty'] = result['penalty']
        hp['C'] = float(result['C'])
        hp['solver'] = result['solver']
        hp['max_iter'] = int(result['max_iter'])
        hp['random_state'] = int(result['random_state'])
        if str(result['class_weight']).lower().strip() == 'none':
            hp['class_weight'] = None
        elif result['class_weight'] == '':
            hp['class_weight'] = {label : float(result['class_weight_' + str(label)]) for label in labels}
        else:
            hp['class_weight'] = result['class_weight']
    elif model=='knn':
        hp['n_neighbors'] = int(result['n_neighbors'])
        hp['weights'] = result['weights']
        hp['algorithm'] = result['algorithm']
        hp['p'] = int(result['p'])
    elif model=='Decision Tree':
        hp['criterion'] = result['criterion']
        hp['splitter'] = result['splitter']
        hp['max_depth'] = None if int(result['max_depth']) == 0 else int(result['max_depth'])
        hp['min_samples_split'] = int(result['min_samples_split'])
        hp['min_samples_leaf'] = int(result['min_samples_leaf'])
        hp['max_features'] = None if result['max_features'] == 'none' else result['max_features']
        hp['random_state'] = int(result['random_state'])
        if str(result['class_weight']).lower().strip() == 'none':
            hp['class_weight'] = None
        elif result['class_weight'] == '':
            hp['class_weight'] = {label : float(result['class_weight_' + str(label)]) for label in labels}
        else:
            hp['class_weight'] = result['class_weight']
    elif model=='Support Vector Machine':
        hp['C'] = float(result['C'])
        hp['kernel'] = result['kernel']
        hp['degree'] = int(result['degree'])
        hp['gamma'] = result['gamma']
        hp['random_state'] = int(result['random_state'])
        if str(result['class_weight']).lower().strip() == 'none':
            hp['class_weight'] = None
        elif result['class_weight'] == '':
            hp['class_weight'] = {label : float(result['class_weight_' + str(label)]) for label in labels}
        else:
            hp['class_weight'] = result['class_weight']
    elif model=='Linear Regression':
        hp['fit_intercept'] = True if result['fit_intercept'].lower() == 'true' else False
    elif model=='Polynomial Regression':
        hp['fit_intercept'] = True if result['fit_intercept'].lower() == 'true' else False
        hp['degree'] = int(result['degree'])
    elif model=='Random Forest Regression':
        hp['n_estimators'] = int(result['n_estimators'])
        hp['criterion'] = result['criterion']
        hp['max_depth'] = None if int(result['max_depth']) == 0 else int(result['max_depth'])
        hp['min_samples_split'] = int(result['min_samples_split'])
        hp['min_samples_leaf'] = int(result['min_samples_leaf'])
        hp['max_features'] = result['max_features']
        hp['bootstrap'] = True if result['bootstrap'] == 'True' else False
        hp['random_state'] = int(result['random_state'])
        hp['min_impurity_decrease'] = float(result['min_impurity_decrease'])
    return hp

def get_sanitize_preprocess_metadata(result):
    result_keys = result.keys()
    ids = [x.split('_')[-1] for x in result_keys if 'column_name_' in x]
    col_name =[]
    func_name =[]
    for col_id in ids:
        col_name.append(result['column_name_' + col_id])
        func_type = result['function_type_' + col_id]
        func_name.append([func_type])
        if 'option_' + col_id in result:
            func_name[-1].append(result['option_' + col_id])
        

    preprocess_steps = list(zip(col_name, func_name))
    return preprocess_steps


def get_sanitize_processed_data(df, preprocess_steps):
    ppdata = sanitize_preprossessteps(df)
    for column, func_option in preprocess_steps:
        if func_option[0] == 'label_encoding':
            ppdata.lable_encoding(column=column)

        # elif func_option[0] == 'fill_na':
        #     ppdata.fill_na(column=column, value=func_option[1])

        elif func_option[0] == 'by_ip':
            ppdata.create_ip(column=column,value=func_option[1])

        elif func_option[0] == 'by_username':
            ppdata.username(column=column)

        elif func_option[0] == 'drop_column':
            ppdata.drop_column(column=column)
        
        elif func_option[0] == 'text':
            ppdata.random_text(column=column)

        elif func_option[0] == 'number':
            ppdata.random_number(column=column)

        # elif func_option[0] == 'pivot_table':
        #     ppdata.pivot_table(column=column,value1=func_option[1],value2=func_option[2],value3=func_option[3])
    df = ppdata.get_df()
    return df