'''
利用 tf.estimator 作预测
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil




'''
数据读取和分布
'''
# 数据加载
# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
LABEL_COLUMN = 'fare_amount'
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

'''
读取方式1： 使用pd读取，这种方式不可取，因为会读取全部的数据
'''
# # 这种方式不和谐： 因为不是构建图的方式，会将数据都读入内存
# # df_train = pd.read_csv('../data/taxi-train.csv', header = None, names = CSV_COLUMNS)
# # df_valid = pd.read_csv('../data/taxi-valid.csv', header = None, names = CSV_COLUMNS)
 #
#
# # 获得训练和验证数据
# def make_input_fn(df, num_epochs):
#   return tf.estimator.inputs.pandas_input_fn(
#     x = df,
#     y = df[LABEL],
#     batch_size = 128,
#     num_epochs = num_epochs,
#     shuffle = True,
#     queue_capacity = 1000,
#     num_threads = 1
#   )
#
# # 获得测试数据
# def make_prediction_input_fn(df, num_epochs):
#   return tf.estimator.inputs.pandas_input_fn(
#     x = df,
#     y = None,
#     batch_size = 128,
#     num_epochs = num_epochs,
#     shuffle = True,
#     queue_capacity = 1000,
#     num_threads = 1
#   )
#
# # 特征
# def make_feature_cols():
#   input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
#   return input_columns




'''
读取方式2： 使用tf.DataSet读取，先构建图，然后每次读取部分数据
'''
# 和谐方式
def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename)
        # Read lines from text files
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        dataset = textlines_dataset.map(decode_csv)

        # Note:
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()

    return _input_fn

def get_train():
  return read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
  return read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL)

def get_test():
  return read_dataset('./taxi-test.csv', mode = tf.estimator.ModeKeys.EVAL)

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

def add_more_features(feats):
  # Nothing to add (yet!)
  return feats

feature_cols = add_more_features(INPUT_COLUMNS)




'''
部署到服务
'''
# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.
def serving_input_fn():
  feature_placeholders = {
    'pickuplon' : tf.placeholder(tf.float32, [None]),
    'pickuplat' : tf.placeholder(tf.float32, [None]),
    'dropofflat' : tf.placeholder(tf.float32, [None]),
    'dropofflon' : tf.placeholder(tf.float32, [None]),
    'passengers' : tf.placeholder(tf.float32, [None]),
  }
  # You can transforma data here from the input format to the format expected by your model.
  features = feature_placeholders # no transformation needed
  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)



# 日志
tf.logging.set_verbosity(tf.logging.INFO)





'''
metrics
'''
# 验证
def print_rmse(model, name, df):
    metrics = model.evaluate(input_fn=make_input_fn(df, 1))
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))





'''
模型
'''
# 模型保存目录
OUTDIR = '../model/taxi_trained'
# 清空目录
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

# 利用单层回归模型做预测
def regreesion_sigle_layer_model():
    # 回归器
    model = tf.estimator.LinearRegressor(feature_columns=make_feature_cols(), model_dir=OUTDIR)
    # 训练
    model.train(input_fn=make_input_fn(df_train, num_epochs=10))

    print_rmse(model, 'validation', df_valid)
    # 测试
    predictions = model.predict(input_fn=make_prediction_input_fn(df_valid, 1))
    for i in range(5):
        print(predictions.next())
    pass

# 利用多层回归模型做预测
def regreesion_multi_layer_model():
    shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
    model = tf.estimator.DNNRegressor(hidden_units=[32, 8, 2], feature_columns=make_feature_cols(), model_dir=OUTDIR)
    model.train(input_fn=make_input_fn(df_train, num_epochs=100))
    print_rmse(model, 'validation', df_valid)
    pass

# 利用estimate API完成训练和评估
def train_and_evaluate(output_dir, num_train_steps):
  estimator = tf.estimator.LinearRegressor(model_dir = output_dir, feature_columns = feature_cols)
  train_spec=tf.estimator.TrainSpec(input_fn = read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN),
                                    max_steps = num_train_steps)
  # 部署的时候使用
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec=tf.estimator.EvalSpec(input_fn = read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL),
                                  steps = None, start_delay_secs = 1, throttle_secs = 10, exporters = exporter)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


