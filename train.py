import pickle
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
from sklearn.linear_model import LogisticRegression


flags.DEFINE_string('input', 'train_data.csv', 'Input File Path which contains training data')
flags.DEFINE_string('output', 'classification_model.pkl', 'Path to output pickel file')

def main(_argv):
    filepath = FLAGS.input
    outpath = FLAGS.output
    
    df = pd.read_csv(filepath)
    
    x_train = df[['id', 'run_id']]
    y_train = df['image_id']
    
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train)

    with open(outpath, 'wb') as f:
        pickle.dump(model, f)
        
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass