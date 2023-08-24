from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt


class ML_model_training:
    def __init__(self, train_images, train_labels, 
                 val_images, val_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
    
    def knn_model(self):
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(self.train_images, self.train_labels)
        knn_predictions = knn.predict(self.val_images)
        knn_accuracy = accuracy_score(self.val_labels, 
                                      knn_predictions)
        print("KNN Accuracy:", knn_accuracy)
        return knn
    
    def xgb_classifier(self):
        xgb = XGBClassifier(n_estimators=2, max_depth=3, 
                            n_jobs=-1, learning_rate=0.01, 
                            random_state=42)
        xgb.fit(self.train_images, self.train_labels)
        xgb_predictions = xgb.predict(self.val_images)
        xgb_accuracy = accuracy_score(self.val_labels, 
                                      xgb_predictions)
        print("XGBoost Accuracy:", xgb_accuracy)
        return xgb
    
    def random_forest_classifier(self):
        rf = RandomForestClassifier(n_estimators=3, max_depth=5, 
                                    n_jobs=-1, random_state=42)
        rf.fit(self.train_images, self.train_labels)
        rf_predictions = rf.predict(self.val_images)
        rf_accuracy = accuracy_score(self.val_labels, rf_predictions)
        print("Random Forest Accuracy:", rf_accuracy)
        return rf
    
    def decision_tree_classifier(self):
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(self.train_images, self.train_labels)
        dt_predictions = dt.predict(self.val_images)
        dt_accuracy = accuracy_score(self.val_labels, dt_predictions)
        print("Decision Tree Accuracy:", dt_accuracy)
        return dt
    
    def logistic_regression(self):
        logreg = LogisticRegression(max_iter=15, n_jobs=-1)
        logreg.fit(self.train_images, self.train_labels)
        logreg_predictions = logreg.predict(self.val_images)
        logreg_accuracy = accuracy_score(self.val_labels, logreg_predictions)
        print("Logistic Regression Accuracy:", logreg_accuracy)
        return logreg
    
    def support_vector_classifier(self):
        svm = SVC()
        svm.fit(self.train_images, self.train_labels)
        svm_predictions = svm.predict(self.val_images)
        svm_accuracy = accuracy_score(self.val_labels, svm_predictions)
        print("SVM Accuracy:", svm_accuracy)
        return svm
    
    def ml_main(self):
        knn = self.knn_model()
        xgb = self.xgb_classifier()
        rf = self.random_forest_classifier()
        dt = self.decision_tree_classifier()
        logreg = self.logistic_regression()
        svm = self.support_vector_classifier()
        models = [knn, xgb, rf, dt, logreg, svm]
        model_names = ['knn', 'xgb', 'rf', 'dt', 'logreg', 'svm']
        models_dict = {key: value for key, value in zip(model_names, models)}
        return models_dict






class DL_model_training:
    def __init__(self, num_classes, train_images, train_labels, 
                 val_images, val_labels, epochs, batch_size,
                 target_img_width, target_img_height, 
                 target_img_channels,  
                 metrics=['accuracy']):
        
        self.num_classes = num_classes
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.target_img_width = target_img_width 
        self.target_img_height = target_img_height
        self.target_img_channels = target_img_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = metrics
        
    def plot_train_history(self, figsize=(6,6)):
        train_accuracy = self.history.history['accuracy']
        train_loss = self.history.history['loss']
        # Create an array of epoch numbers
        EPOCHS = range(1, len(train_accuracy) + 1)
        plt.figure(figsize=figsize)
        plt.plot(EPOCHS, train_accuracy, 'b.-', label='Train_Accuracy')
        plt.plot(EPOCHS, train_loss, 'r.-', label='Train_Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
            
    def cnn_model_checkpoint(self, checkpoint_filepath):  
        checkpoint_filepath = checkpoint_filepath
        checkpoint = ModelCheckpoint(checkpoint_filepath, 
                                    monitor='accuracy',
                                    save_best_only=True, 
                                    mode='max', 
                                    verbose=1)
        return checkpoint
    
    def cnn_model_early_stopping(self, patience=20):
        early_stopping = EarlyStopping(monitor='accuracy', 
                                patience=patience, 
                                baseline=90,
                                restore_best_weights=True)
        return early_stopping
    
    def Data_Augmentation(self):
        # Define the augmentation parameters
        datagen = ImageDataGenerator(
        #     rotation_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
        #     brightness_range=[0.8, 1.2],
            fill_mode='nearest')
        train_generator = datagen.flow(self.train_images, 
                                       self.train_labels, 
                                       self.batch_size)
        return train_generator
            
    def cnn_model(self):
        # Create the model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(self.target_img_height, 
                                self.target_img_width, 
                                self.target_img_channels)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        cnn_model.compile(optimizer='adam', 
                          loss='sparse_categorical_crossentropy', 
                          metrics=self.metrics)
        return cnn_model
    
    def model_training(self, model, model_filepath, 
                       early_stopping_patience,
                       use_data_augmentation=False, 
                       use_callbacks=True):
        
        callbacks = []
        if use_callbacks:
            callbacks = [
                self.cnn_model_checkpoint(model_filepath),
                self.cnn_model_early_stopping(early_stopping_patience)
            ]
            
        if use_data_augmentation:
            train_generator = self.Data_Augmentation()
            # Train the model
            self.model_history = model.fit(train_generator, epochs=self.epochs, 
                          batch_size=self.batch_size, callbacks=callbacks)
            # Evaluate the model
            model.evaluate(self.val_images, self.val_labels)
            return model
        
        # Train the model
        self.model_history = model.fit(self.train_images, self.train_labels, 
                      epochs=self.epochs, batch_size=self.batch_size,
                      callbacks=callbacks)
        # Evaluate the model
        evaluation_results = model.evaluate(self.val_images, self.val_labels)
        print("Evaluation results:", evaluation_results)
        return model
    
    def transfer_learning(self):
        base_model = MobileNetV2(weights='imagenet', 
                   include_top=False, 
                   input_shape=(self.target_img_height, 
                                self.target_img_width, 
                                self.target_img_channels))

        # Freeze the pre-trained layers so they are not updated during training
        for layer in base_model.layers:
            layer.trainable = False  

        # Add a new classification head on top of the pre-trained layers
        model = keras.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dropout(.2),      
            layers.Dense(units=40, activation='relu'),
            layers.Dense(units=150, activation='relu'),
            layers.Dropout(.2),  
            layers.Dense(units=50, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
            ])

        # Compile the model
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=self.metrics)

        return model
    
