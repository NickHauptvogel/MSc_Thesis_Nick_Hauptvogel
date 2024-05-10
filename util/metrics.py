import keras

if keras.__version__  >= '3.0.0':
    def f1_score(y_true, y_pred, eps=1e-7):
        true_positives = keras.ops.sum(keras.ops.round(keras.ops.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.ops.sum(keras.ops.round(keras.ops.clip(y_true, 0, 1)))
        predicted_positives = keras.ops.sum(keras.ops.round(keras.ops.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + eps)
        recall = true_positives / (possible_positives + eps)
        f1_val = 2*(precision*recall) / (precision+recall + eps)
        return f1_val
else:
    import keras.backend as K
    def f1_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val