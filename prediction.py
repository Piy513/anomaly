import pickle
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('input', 'train_data.csv', 'Input file path which contains data for prediction')
flags.DEFINE_string('model', 'classification_model.pkl', 'Path to trained model file')
flags.DEFINE_string('output', 'output.csv', 'Output file path')

def main(_argv):
    filepath = FLAGS.input
    pickle_file = FLAGS.model
    outpath = FLAGS.output

    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)
    
    df_test = pd.read_csv(filepath)
    
    x_test = df_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    df_test['prediction'] = model.predict(x_test)

    df_test.to_csv(outpath, index=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass