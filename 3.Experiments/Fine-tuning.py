import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFElectraForSequenceClassification, TFBertForSequenceClassification, TFAlbertForSequenceClassification, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
set_seed(42)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs {gpus} are available and set for memory growth.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs are available.")

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if num_classes is None:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def load_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        docs = [doc.strip().split('\t') for doc in f]
        docs = [(doc[0], int(doc[1])) for doc in docs if len(doc) == 2]
        texts, labels = zip(*docs)
        y_one_hot = to_categorical(labels)
        return train_test_split(texts, y_one_hot, test_size=0.2, random_state=0)
    
class CustomTFBertForSequenceClassification(TFBertForSequenceClassification):
    @staticmethod
    def custom_unpack_x_y_sample_weight(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data
        return data, None, None

    def train_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

class CustomTFElectraForSequenceClassification(TFElectraForSequenceClassification):
    @staticmethod
    def custom_unpack_x_y_sample_weight(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data
        return data, None, None

    def train_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

class CustomTFAlbertForSequenceClassification(TFAlbertForSequenceClassification):
    @staticmethod
    def custom_unpack_x_y_sample_weight(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data
        return data, None, None

    def train_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

class CustomTFRobertaForSequenceClassification(TFRobertaForSequenceClassification):
    @staticmethod
    def custom_unpack_x_y_sample_weight(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data
        return data, None, None

    def train_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

class CustomTFKcBertForSequenceClassification(TFBertForSequenceClassification):
    @staticmethod
    def custom_unpack_x_y_sample_weight(data):
        if isinstance(data, tuple):
            if len(data) == 2:
                return data[0], data[1], None
            elif len(data) == 3:
                return data
        return data, None, None

    def train_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = self.custom_unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

train_file = 'train_data.txt'

X_train, X_test, y_train, y_test = load_data(train_file)

def train_and_evaluate(model_name, model_class, tokenizer_name, batch_size, learning_rate, max_length):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = model_class.from_pretrained(model_name, num_labels=2, from_pt=True)
    
    X_train_tokenized = tokenizer(X_train, return_tensors="tf", max_length=max_length, padding='max_length', truncation=True)
    X_test_tokenized = tokenizer(X_test, return_tensors="tf", max_length=max_length, padding='max_length', truncation=True)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train_tokenized), y_train)).shuffle(len(y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test_tokenized), y_test)).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    checkpoint_filepath = f"./checkpoints/checkpoint_{model_name.replace('/', '_')}_maxlen_{max_length}"
    mc = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', mode='min', 
                                            save_best_only=True, save_weights_only=True)
    

    memory_usage = []

    class MemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            memory_usage.append(memory_info['current'])
    
    memory_callback = MemoryCallback()
    
    history = model.fit(train_dataset, epochs=1000, validation_data=test_dataset, callbacks=[es, mc, memory_callback])
    

    average_memory_usage = np.mean(memory_usage)
    
    model.load_weights(checkpoint_filepath)
    y_preds = model.predict(test_dataset)
    prediction_probs = tf.nn.softmax(y_preds.logits, axis=1).numpy()
    y_predictions = np.argmax(prediction_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    report = classification_report(y_test_labels, y_predictions, output_dict=True)
    accuracy = report['accuracy']
    recall = report['weighted avg']['recall']
    precision = report['weighted avg']['precision']
    f1 = report['weighted avg']['f1-score']
    
    epochs_trained = len(history.history['loss'])
    

    tf.keras.backend.clear_session()
    tf.config.experimental.reset_memory_stats('GPU:0')
    
    return accuracy, recall, precision, f1, epochs_trained, average_memory_usage

results = []

model_configs = [
    {"model_name": "kykim/albert-kor-base", "model_class": CustomTFAlbertForSequenceClassification, "tokenizer_name": "kykim/albert-kor-base"},
    {"model_name": "klue/bert-base", "model_class": CustomTFBertForSequenceClassification, "tokenizer_name": "klue/bert-base"},
    {"model_name": "klue/roberta-base", "model_class": CustomTFRobertaForSequenceClassification, "tokenizer_name": "klue/roberta-base"},
    {"model_name": "beomi/kcbert-base", "model_class": CustomTFKcBertForSequenceClassification, "tokenizer_name": "beomi/kcbert-base"},
    {"model_name": "beomi/KcELECTRA-base-v2022", "model_class": CustomTFElectraForSequenceClassification, "tokenizer_name": "beomi/KcELECTRA-base-v2022"}
]

batch_size = 16
learning_rate = 2e-5
max_lengths = [16, 32, 64, 128]


for config in model_configs:
    for max_length in max_lengths:
        accuracy, recall, precision, f1, epochs_trained, average_memory_usage = train_and_evaluate(
            config["model_name"], 
            config["model_class"], 
            config["tokenizer_name"], 
            batch_size, 
            learning_rate, 
            max_length
        )
        results.append((config["model_name"], max_length, accuracy, recall, precision, f1, epochs_trained, average_memory_usage / 1024))  # Convert to MB

results_df = pd.DataFrame(results, columns=["Model Name", "Max Length", "Accuracy", "Recall", "Precision", "F1-Score", "Epochs Trained", "Average Memory Usage (MB)"])
results_df.to_csv('model_comparison_memory_usage3.csv', index=False)
print(results_df)
