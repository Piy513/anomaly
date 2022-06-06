import re
import pickle
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
from sklearn.ensemble import RandomForestClassifier


flags.DEFINE_string('input', 'train_data.csv', 'Input File Path which contains training data')
flags.DEFINE_string('output', 'classification_model.pkl', 'Path to output pickel file')

def main(_argv):
    filepath = FLAGS.input
    outpath = FLAGS.output
    
    df = pd.read_csv(filepath)
    
    x_train = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y_train = df['Species']
    
    model = RandomForestClassifier(**{'n_estimators': 164, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True, 'random_state': 42, 'class_weight': None})
    model.fit(x_train, y_train)

    with open(outpath, 'wb') as f:
        pickle.dump(model, f)
        
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass