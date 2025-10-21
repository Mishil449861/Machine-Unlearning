import tensorflow as tf
import numpy as np
from tqdm import tqdm

def build_model(input_dim, num_classes, dropout_rate=0.3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_sisa_models(x_train, y_train, num_shards):
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    shards = np.array_split(indices, num_shards)
    models = []
    for shard in tqdm(shards, desc="Training SISA Shards"):
        X_shard, y_shard = x_train[shard], y_train[shard]
        model = build_model(x_train.shape[1], len(np.unique(y_train)))
        model.fit(X_shard, y_shard, epochs=10, batch_size=128, verbose=0, validation_split=0.1)
        models.append(model)
    return models, shards

def unlearn_class(models, shards, x_train, y_train, forget_class):
    for i, shard in enumerate(shards):
        mask = y_train[shard] != forget_class
        if np.sum(mask) < len(y_train[shard]):
            X_shard_new, y_shard_new = x_train[shard][mask], y_train[shard][mask]
            models[i] = build_model(x_train.shape[1], len(np.unique(y_train)))
            models[i].fit(X_shard_new, y_shard_new, epochs=5, batch_size=128, verbose=0)
    return models

def get_ensemble_predictions(models, X):
    """
    Combines predictions from all models in the ensemble by averaging their softmax outputs.
    Returns the predicted class labels.
    """
    preds = []

    for model in models:
        y_prob = model.predict(X, verbose=0)
        preds.append(y_prob)

    # Average probabilities across ensemble members
    avg_pred = np.mean(preds, axis=0)
    y_pred = np.argmax(avg_pred, axis=1)
    return y_pred
