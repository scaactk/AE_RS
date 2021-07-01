import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow.keras.backend as K
import warnings

warnings.filterwarnings('ignore')

# 读取评分数据
df = pd.read_csv('ml1m_ratings.csv', sep='\t', encoding='latin-1',
                 usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

# 获得用户物品数量，不能直接这么用，，编号不连续，容易翻车，除非重建索引编号
# num_users = df['user_emb_id'].nunique()
# num_movies = df['movie_emb_id'].nunique()
num_users = df['user_emb_id'].unique().max() + 1
num_movies = df['movie_emb_id'].unique().max() + 1

# 数据集划分，按user_emb_id的在比例划分
train_df, test_df = model_selection.train_test_split(df, stratify=df['user_emb_id'], test_size=0.1, random_state=1234)
train_df, validate_df = model_selection.train_test_split(train_df, stratify=train_df['user_emb_id'], test_size=0.1,
                                                         random_state=999613182)


def dataPreprocessor(rating_df, num_users, num_items, init_value=0, average=False):
    # 用户总数和物品总数在构建子矩阵的时候保持不变，即为每个子矩阵，都一样大，方便后面对应计算
    # 返回构建好的矩阵
    if average:
        matrix = np.full((num_users, num_items), 0.0)
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating
        # average = 每一行的总合 / 该行不为零的个数
        line_number = (matrix != 0).sum(1)  # 该行不为零的个数，前面判断后返回真值矩阵，后面计算个数
        average = np.true_divide(matrix.sum(1), np.maximum(line_number, 1))  # 防止除数为0
        indxs = np.where(matrix == 0)  # 取出索引（得是matrix才行，list不行）
        matrix[indxs] = np.take(average, indxs[0])  # 索引按行，该行0值都用该行的average

    else:
        matrix = np.full((num_users, num_items), float(init_value))
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating

    return matrix


# 获取需要的矩阵
users_items_matrix_train_average = dataPreprocessor(train_df, num_users, num_movies, average=True)
users_items_matrix_train_zero = dataPreprocessor(train_df, num_users, num_movies, 0)

users_items_matrix_validate = dataPreprocessor(validate_df, num_users, num_movies, 0)
users_items_matrix_test = dataPreprocessor(test_df, num_users, num_movies, 0)


def masked_se(y_true, y_pred):
    mask = K.not_equal(y_true, 0)  # 返回布尔值
    mask = K.cast_to_floatx(mask)
    loss = K.square(mask * (y_true - y_pred))
    loss = K.sum(loss, axis=-1)
    return loss


def masked_rmse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    loss = K.square(mask * (y_true - y_pred))
    loss = K.sum(loss, axis=-1) / K.maximum(K.sum(mask, axis=-1), 1)
    loss = K.sqrt(loss)
    return loss


def masked_rmse_clip(y_true, y_pred):
    mask = K.not_equal(y_true, 0)
    mask = K.cast_to_floatx(mask)
    y_pred = K.clip(y_pred, 1, 5)  # 将超出范围的限制在其中
    loss = K.square(mask * (y_true - y_pred))
    loss = K.sum(loss, axis=-1) / K.maximum(K.sum(mask, axis=-1), 1)
    return loss


def AutoRec(shape, reg):
    input = keras.Input(shape=shape)
    x = layers.Dense(500, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(reg))(input)
    output = layers.Dense(shape, activation='elu', kernel_regularizer=keras.regularizers.L2(reg))(x)

    model = keras.Model(input, output)
    return model


test_auto_rec = AutoRec(users_items_matrix_train_average.shape[1], 0.0005)
test_auto_rec.compile(optimizer=keras.optimizers.Adam(0.0001), loss=masked_rmse, metrics=[masked_rmse_clip])
test_auto_rec.summary()

test_auto_rec.fit(users_items_matrix_train_average, users_items_matrix_train_zero, epochs=500, batch_size=256,
                  validation_data=(users_items_matrix_train_average, users_items_matrix_validate))

