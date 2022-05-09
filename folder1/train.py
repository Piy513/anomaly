import re
import math
import pickle
import numpy as np
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('input', 'train_data.csv', 'Input File Path which contains training data')
flags.DEFINE_string('output', 'trained_model.pkl', 'Path to output pickel file')
flags.DEFINE_float('k_value','2.5','k value for setting threshold')

def find_threshold(values, k):
    vals = pd.Series(values)
    mean = vals.mean()
    std = vals.std()
    if pd.isna(std):
        std = 0
    return math.ceil(mean + k*std)

def main(_argv):
    filepath = FLAGS.input
    outpath = FLAGS.output
    k_value = FLAGS.k_value
    
    df = pd.read_csv(filepath)
    
    df = df.groupby(['image_id', 'file_name']).agg({'image_id': 'nunique'}).reset_index()
    
    df_group = df.groupby(['image_id']).agg(image_id_hist = ('image_id', list)).reset_index()

    for column in ['image_id']:
        new_column_name = re.sub(r'[^a-zA-Z0-9_]', '_', column)
        if new_column_name[0].isalpha() or new_column_name[0] == '_':
            pass
        else:
            new_column_name = '_' + new_column_name
        df_group[column + '_threshold'] = df_group[new_column_name + '_hist'].apply(find_threshold, k=k_value)

    data_to_save = {}
    data_to_save['df_group'] = df_group
    
    with open(outpath, 'wb') as f:
        pickle.dump(data_to_save, f)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass