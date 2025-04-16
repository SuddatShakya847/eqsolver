import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pickle

class MathSymbolClassifier:
    def __init__(self, model_path=None):
        """Initialize the classifier."""
        self.model = None
        self.symbol_map = None
        self.input_shape = (28, 28, 1)  # Standard size for symbol images
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
            
    def build_model(self):
        """Build a CNN model for character recognition."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer - will be set during training based on number of classes
            layers.Dense(16, activation='softmax')  # Default to 16 classes, will be updated
        ])
        
        # Using Adam optimizer with a learning rate scheduler
        optimizer = optimizers.Adam(learning_rate=0.001)
        
        model.compile(optimizer=optimizer,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def prepare_dataset(self, data_dir, test_size=0.2, validation_size=0.2):
        """Prepare dataset from a directory of symbol images.
        
        Expected directory structure:
        data_dir/
            0/      # Images of digit 0
                img1.png
                img2.png
                ...
            1/      # Images of digit 1
                ...
            +/      # Images of plus symbol
                ...
            -/      # Images of minus symbol
                ...
            ...
        """
        X = []  # Images
        y = []  # Labels
        symbol_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Create symbol map
        self.symbol_map = {i: symbol for i, symbol in enumerate(sorted(symbol_dirs))}
        reverse_symbol_map = {symbol: i for i, symbol in self.symbol_map.items()}
        
        # Load images and labels
        for symbol_dir in symbol_dirs:
            label = reverse_symbol_map[symbol_dir]
            dir_path = os.path.join(data_dir, symbol_dir)
            
            for img_file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize to standard size
                    img = cv2.resize(img, (28, 28))
                    # Normalize
                    img = img / 255.0
                    X.append(img)
                    y.append(label)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for CNN input
        X = X.reshape(-1, 28, 28, 1)
        
        # Split into train and temporary test set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)
        
        # Split temporary test set into validation and test set
        val_ratio = validation_size / (1 - test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
                                                    random_state=42, stratify=y_temp)
        
        # Update model output layer based on number of classes
        num_classes = len(self.symbol_map)
        self.model.layers[-1] = layers.Dense(num_classes, activation='softmax')
        self.model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train the model on the prepared dataset."""
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Setup callbacks
        checkpoint = ModelCheckpoint('best_math_symbols_model.h5', 
                                    monitor='val_accuracy', 
                                    save_best_only=True, 
                                    mode='max', 
                                    verbose=1)
        
        early_stopping = EarlyStopping(monitor='val_accuracy', 
                                      patience=10, 
                                      restore_best_weights=True, 
                                      verbose=1)
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save the symbol map
        with open('symbol_map.pkl', 'wb') as f:
            pickle.dump(self.symbol_map, f)
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_acc:.4f}')
        return test_acc
    
    def predict(self, image):
        """Predict the class of a single image."""
        if image.shape != self.input_shape:
            # Resize and normalize if needed
            image = cv2.resize(image, (28, 28))
            
            # Ensure image is normalized to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
                
            # Add batch and channel dimensions if needed
            if len(image.shape) == 2:
                image = image.reshape(1, 28, 28, 1)
        
        # Get prediction
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction)
        
        # Return symbol using the symbol map
        if self.symbol_map and predicted_class in self.symbol_map:
            return self.symbol_map[predicted_class]
        else:
            return str(predicted_class)
    
    def save_model(self, model_path='math_symbols_model.h5', map_path='symbol_map.pkl'):
        """Save the model and symbol map."""
        self.model.save(model_path)
        
        with open(map_path, 'wb') as f:
            pickle.dump(self.symbol_map, f)
    
    def load_model(self, model_path='math_symbols_model.h5', map_path='symbol_map.pkl'):
        """Load a saved model and symbol map."""
        self.model = models.load_model(model_path)
        
        if os.path.exists(map_path):
            with open(map_path, 'rb') as f:
                self.symbol_map = pickle.load(f)
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def create_synthetic_dataset(self, output_dir, samples_per_class=1000):
        """Create a synthetic dataset for training if real data is not available."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the symbols we want to recognize
        symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  '+', '-', '*', '/', '=', 'x']
        
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, 
                cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
        
        for symbol in symbols:
            # Create directory for this symbol
            symbol_dir = os.path.join(output_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                # Create a blank image
                img = np.ones((64, 64), dtype=np.uint8) * 255
                
                # Choose a random font
                font = fonts[np.random.randint(0, len(fonts))]
                
                # Random scaling
                scale = 0.8 + np.random.rand() * 1.5
                
                # Random thickness
                thickness = np.random.randint(1, 4)
                
                # Random position with margin
                margin = 5
                x = margin + np.random.randint(0, 64 - 2 * margin)
                y = margin + np.random.randint(0, 64 - 2 * margin) + 16  # +16 to center vertically
                
                # Draw the symbol
                cv2.putText(img, symbol, (x, y), font, scale, 0, thickness)
                
                # Add some noise and distortion
                if np.random.rand() > 0.5:
                    # Add noise
                    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                    img = cv2.add(img, noise)
                
                # Random rotation
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((32, 32), angle, 1)
                img = cv2.warpAffine(img, M, (64, 64))
                
                # Resize to standard 28x28
                img = cv2.resize(img, (28, 28))
                
                # Save the image
                img_path = os.path.join(symbol_dir, f'{i}.png')
                cv2.imwrite(img_path, img)
        
        print(f"Created synthetic dataset with {len(symbols)} classes and {samples_per_class} samples per class")
        return output_dir


# Example usage:
if __name__ == "__main__":
    # Initialize the classifier
    classifier = MathSymbolClassifier()
    
    # Create synthetic dataset if needed
    dataset_dir = "synthetic_math_symbols"
    if not os.path.exists(dataset_dir):
        classifier.create_synthetic_dataset(dataset_dir, samples_per_class=500)
    
    # Prepare the dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = classifier.prepare_dataset(dataset_dir)
    
    # Train the model
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=20)
    
    # Evaluate
    classifier.evaluate(X_test, y_test)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Save the model
    classifier.save_model()
    
    # Test prediction on a sample image
    sample_img = X_test[0]
    prediction = classifier.predict(sample_img)
    print(f"Predicted: {prediction}, Actual: {classifier.symbol_map[y_test[0]]}")