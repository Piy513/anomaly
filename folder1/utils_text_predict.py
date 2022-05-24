class preprossessteps_predict:
    def drop_duplicate(self,column):
        outtextaaaa = "df_test.drop_duplicates(subset=" + str(column) + ", keep='first', inplace=True).reset_index(drop=True)"
        outtext += "\n    "
        outtext += "df_test.reset_index(drop=True, inplace=True)"
        return outtext
            
    def drop_na(self,column):
        outtext = "df_test.dropna(subset=" + str(column) + ", inplace=True).reset_index(drop=True)"
        outtext += "\n    "
        outtext += "df_test.reset_index(drop=True, inplace=True)"
        return outtext
        
    def fill_na(self,column,value=0):
        outtext = ""
        if str(value).lower().strip() in ['mean', 'median', 'mode']:
            outtext = "with open('imputer_" + '_'.join(column) + ".pkl', 'rb') as f:"
            outtext += "\n        "
            outtext += "imputer = pickle.load(f)"
            outtext += "\n    "
            outtext += "df_test[" + str(column) + "] = imputer.transform(df_test[" + str(column) + "])"
        elif type(value) in (int, float):
            return "df_test[" + str(column) + "] = df_test[" + str(column) + "].fillna(" + str(value) + ")"
        return "df_test[" + str(column) + "] = df_test[" + str(column) + "].fillna('" + str(value) + "')"
        
    def lable_encoding(self,column):
        outtext = "with open('label_encoder_" + column + ".pkl', 'rb') as f:"
        outtext += "\n        "
        outtext += "label_encoder = pickle.load(f)"
        outtext += "\n    "
        outtext += "df_test['" + column + "_labels']= label_encoder.transform(df_test['" + column + "'])"
        return outtext
    
    def one_hot_encode(self,column):
        outtext = "with open('onehot_encoder_" + '_'.join(column) + ".pkl', 'rb') as f:"
        outtext += "\n        "
        outtext += "onehot_encoder = pickle.load(f)"
        outtext += "\n    "
        outtext += "onehot_encoded = onehot_encoder.transform(df_test[" + str(column) + "])"
        outtext += "\n    "
        outtext += "tokens = onehot_encoder.get_feature_names(input_features=" + str(column) + ")"
        outtext += "\n    "
        outtext += "df_onehot = pd.DataFrame(data = onehot_encoded.toarray(),columns = tokens)"
        outtext += "\n    "
        outtext += "df_onehot = df_onehot.set_index(df_test.index)"
        outtext += "\n    "
        outtext += "df_test = pd.concat([df_test,df_onehot],axis=1)"
        return outtext
    
    def tfidf_vectorization(self,column):
        outtext = "with open('tfidf_vectorizer_" + column + ".pkl', 'rb') as f:"
        outtext += "\n        "
        outtext += "vectorizer = pickle.load(f)"
        outtext += "\n    "
        outtext += "vectors = vectorizer.transform(df_test['" + column + "'])"
        outtext += "\n    "
        # outtext += "tokens = vectorizer.get_feature_names()"
        # outtext += "\n    "
        # outtext += "df_vector = pd.DataFrame(data = vectors.toarray(),columns = tokens)"
        # outtext += "\n    "
        # outtext += "df_vector = df_vector.set_index(df_test.index)"
        # outtext += "\n    "
        # outtext += "df_test = pd.concat([df_test,df_vector],axis=1)"
        outtext += "df_test[" + column + " + '_tfidf'] = list(vectors.toarray())"
        return outtext

    def count_vectorization(self,column):
        outtext = "with open('count_vectorizer_" + column + ".pkl', 'rb') as f:"
        outtext += "\n        "
        outtext += "vectorizer = pickle.load(f)"
        outtext += "\n    "
        outtext += "vectors = vectorizer.transform(df_test['" + column + "'])"
        outtext += "\n    "
        # outtext += "tokens = vectorizer.get_feature_names()"
        # outtext += "\n    "
        # outtext += "df_vector = pd.DataFrame(data = vectors.toarray(),columns = tokens)"
        # outtext += "\n    "
        # outtext += "df_vector = df_vector.set_index(df_test.index)"
        # outtext += "\n    "
        # outtext += "df_test = pd.concat([df_test,df_vector],axis=1)"
        outtext += "df_test[" + column + " + '_count_vect'] = list(vectors.toarray())"
        return outtext
    
    def  MinMaxScaler(self,column):
        outtext = "with open('minmax_scaler_" + '_'.join(column) + ".pkl', 'rb') as f:"
        outtext += "\n        "
        outtext += "scaler = pickle.load(f)"
        outtext += "\n    "
        outtext += "scaled_data = scaler.transform(df_test[" + str(column) + "])"
        outtext += "\n    "
        outtext += "tokens = " + str([col+'_minmaxscaled' for col in column])
        outtext += "\n    "
        outtext += "df_minmax = pd.DataFrame(data = scaled_data, columns = tokens)"
        outtext += "\n    "
        outtext += "df_minmax = df_minmax.set_index(df_test.index)"
        outtext += "\n    "
        outtext += "df_test = pd.concat([df_test,df_minmax],axis=1)"
        return outtext

    def split_to_list(self,column, split_pattern):
        if split_pattern != '':
            outtext = "df_test['" + column + "_list'] = df_test['" + column + "'].apply(lambda x: str(x).strip().split('" + split_pattern + "'))"
        else:
            outtext = "df_test['" + column + "_list'] = df_test['" + column + "'].apply(lambda x: str(x).strip().split())"
        return outtext

    def filter(self,column, values):
        return "df_test = df_test[df_test['" + column + "'].isin('" + str(values) + "')]"