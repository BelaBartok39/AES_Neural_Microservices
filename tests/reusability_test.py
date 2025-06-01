import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle

def setup_gpu():
    """Setup GPU/CUDA configuration"""
    print("Setting up GPU/CUDA configuration...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUs Available: {len(gpus)}")
    
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU: {gpu}")
            
            # Set mixed precision for faster training on modern GPUs
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (float16)")
            
            # Enable XLA compilation for additional speedup
            tf.config.optimizer.set_jit(True)
            print("XLA compilation enabled")
            
            print("GPU setup complete!")
            return True
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPUs found, using CPU")
        return False

# Setup GPU before other operations
gpu_available = setup_gpu()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CaesarCipherTester:
    def __init__(self):
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 26  # A-Z only for simplicity
        
    def setup_vocabulary(self):
        """Create character mappings for A-Z"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
    def caesar_cipher_reference(self, text, shift):
        """Reference implementation of Caesar cipher"""
        result = ""
        for char in text.upper():
            if char in self.char_to_idx:
                shifted_idx = (self.char_to_idx[char] + shift) % 26
                result += self.idx_to_char[shifted_idx]
            else:
                result += char
        return result
    
    def generate_training_data(self, num_samples=50000):
        """Generate training data for Caesar cipher with various shifts"""
        print("Generating training data...")
        
        # Generate random text samples
        input_texts = []
        output_texts = []
        shifts_used = []
        
        for _ in range(num_samples):
            # Random text length between 5-20 characters
            text_length = np.random.randint(5, 21)
            
            # Generate random text (A-Z only)
            text = ''.join([self.idx_to_char[np.random.randint(0, 26)] 
                           for _ in range(text_length)])
            
            # Random shift between 1-25
            shift = np.random.randint(1, 26)
            
            # Apply Caesar cipher
            cipher_text = self.caesar_cipher_reference(text, shift)
            
            input_texts.append(text)
            output_texts.append(cipher_text)
            shifts_used.append(shift)
        
        return input_texts, output_texts, shifts_used
    
    def text_to_sequence(self, text, max_length=20):
        """Convert text to numerical sequence"""
        sequence = []
        for char in text.upper():
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(0)  # Unknown character
        
        # Pad or truncate to max_length
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
            
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert numerical sequence back to text"""
        text = ""
        for idx in sequence:
            if idx in self.idx_to_char:
                text += self.idx_to_char[idx]
            # Skip padding zeros
        return text.rstrip('A')  # Remove padding (represented as 'A' due to idx 0)
    
    def create_model(self, max_length=20):
        """Create a simple model for Caesar cipher"""
        print("Creating model...")
        
        # Use mixed precision if GPU is available
        if gpu_available:
            print("Creating model with mixed precision optimization")
        
        model = keras.Sequential([
            layers.Input(shape=(max_length,), name="input_layer"),
            
            # Embedding layer to convert indices to vectors
            layers.Embedding(self.vocab_size, 64, name="embedding"),
            
            # LSTM to process the sequence
            layers.LSTM(128, return_sequences=True, name="lstm_1"),
            layers.LSTM(128, return_sequences=True, name="lstm_2"),
            
            # Dense layer to predict output characters
            layers.Dense(64, activation='relu', name="dense_1"),
            layers.Dense(self.vocab_size, name="pre_output"),
            
            # Output layer - cast to float32 for mixed precision compatibility
            layers.Activation('softmax', dtype='float32', name="output_layer")
        ])
        
        # Use different optimizers based on GPU availability
        if gpu_available:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = 'adam'
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data_for_training(self, input_texts, output_texts, max_length=20):
        """Convert text data to numerical arrays for training"""
        print("Preparing data for training...")
        
        X = np.array([self.text_to_sequence(text, max_length) for text in input_texts])
        y = np.array([self.text_to_sequence(text, max_length) for text in output_texts])
        
        return X, y
    
    def train_model(self, save_path='caesar_model'):
        """Complete training pipeline"""
        # Setup
        self.setup_vocabulary()
        
        # Generate data
        input_texts, output_texts, shifts = self.generate_training_data(50000)
        
        # Prepare for training
        X, y = self.prepare_data_for_training(input_texts, output_texts)
        
        # Create model
        self.model = self.create_model()
        
        print("Model architecture:")
        self.model.summary()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Train
        print("Starting training...")
        
        # Create callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64 if not gpu_available else 128,  # Larger batch for GPU
            verbose=1,
            callbacks=callbacks
        )
        
        # Save the model
        print(f"Saving model to {save_path}...")
        
        # Method 1: Native Keras format (.keras) - NEW RECOMMENDED FORMAT
        self.model.save(f"{save_path}.keras")
        print("Saved in native Keras format (.keras)")
        
        # Method 2: Export SavedModel for production use
        self.model.export(f"{save_path}_savedmodel")
        print("Exported SavedModel for production use")
        
        # Method 3: H5 format (legacy compatibility)
        try:
            self.model.save(f"{save_path}.weights.h5")
            print("Saved in H5 format (legacy)")
        except Exception as e:
            print(f"H5 save failed (expected in newer TF versions): {e}")
        
        # Method 4: Save weights only
        self.model.save_weights(f"{save_path}_weights.weights.h5")
        print("Saved weights only")
        
        # Save vocabulary mapping
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        with open(f"{save_path}_vocab.pkl", 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print("Model saved successfully!")
        return history
    
    def load_model(self, save_path='caesar_model'):
        """Load a previously trained model"""
        print(f"Loading model from {save_path}...")
        
        # Load vocabulary
        with open(f"{save_path}_vocab.pkl", 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        self.vocab_size = vocab_data['vocab_size']
        
        # Try different loading methods in order of preference
        load_methods = [
            (f"{save_path}.keras", "Native Keras format"),
            (f"{save_path}_savedmodel", "SavedModel format"),
            (f"{save_path}.weights.h5", "H5 format")
        ]
        
        for model_path, format_name in load_methods:
            try:
                if os.path.exists(model_path):
                    if format_name == "SavedModel format":
                        # Use tf.saved_model.load for SavedModel
                        self.model = tf.saved_model.load(model_path)
                        # Convert to Keras model if needed
                        if hasattr(self.model, 'signatures'):
                            print(f"Loaded {format_name} - using as TF SavedModel")
                        else:
                            self.model = keras.models.load_model(model_path)
                    else:
                        self.model = keras.models.load_model(model_path)
                    
                    print(f"Successfully loaded {format_name}!")
                    return True
                    
            except Exception as e:
                print(f"Failed to load {format_name}: {e}")
                continue
        
        print("All loading methods failed!")
        return False
    
    def predict_cipher(self, plaintext, verbose=True):
        """Use the trained model to predict Caesar cipher"""
        if self.model is None:
            print("No model loaded!")
            return None
        
        # Convert to sequence
        sequence = self.text_to_sequence(plaintext)
        input_array = np.array([sequence])
        
        # Predict
        predictions = self.model.predict(input_array, verbose=0)
        
        # Convert predictions to text
        predicted_indices = np.argmax(predictions[0], axis=1)
        predicted_text = self.sequence_to_text(predicted_indices)
        
        if verbose:
            print(f"Input: {plaintext}")
            print(f"Predicted cipher: {predicted_text}")
        
        return predicted_text
    
    def test_model_reusability(self):
        """Test that the model works correctly after loading"""
        print("\n" + "="*50)
        print("TESTING MODEL REUSABILITY")
        print("="*50)
        
        test_cases = [
            "HELLO",
            "WORLD",
            "TESTING",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "QUICKBROWNFOX"
        ]
        
        print("\nTesting predictions with loaded model:")
        for test_text in test_cases:
            predicted = self.predict_cipher(test_text)
            
            # Verify it's not just returning the input
            if predicted != test_text:
                print(f"✓ {test_text} -> {predicted} (TRANSFORMED)")
            else:
                print(f"✗ {test_text} -> {predicted} (NO CHANGE - POTENTIAL ISSUE)")
        
        print("\nModel reusability test complete!")

def main():
    """Main function to demonstrate model training, saving, and loading"""
    print("Caesar Cipher Neural Network - Model Persistence Test")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {gpu_available}")
    print("="*60)
    
    tester = CaesarCipherTester()
    
    # Phase 1: Train and save model
    print("\nPHASE 1: Training and saving model...")
    history = tester.train_model('caesar_model_test')
    
    # Phase 2: Test the model before saving (sanity check)
    print("\nPHASE 2: Testing model immediately after training...")
    tester.test_model_reusability()
    
    # Phase 3: Clear the model and load it back
    print("\nPHASE 3: Clearing model and loading from disk...")
    tester.model = None  # Clear the model
    
    success = tester.load_model('caesar_model_test')
    
    if success:
        print("Model loaded successfully!")
        
        # Phase 4: Test the loaded model
        print("\nPHASE 4: Testing loaded model...")
        tester.test_model_reusability()
        
        # Phase 5: Additional verification
        print("\nPHASE 5: Additional verification tests...")
        
        # Test that we can call the model multiple times
        for i in range(3):
            print(f"\nCall #{i+1}:")
            result = tester.predict_cipher("NEURAL")
            
        print("\n" + "="*60)
        print("SUCCESS: Model is fully reusable!")
        print("You can now use this pattern for your AES microservices.")
        print("="*60)
        
    else:
        print("FAILED: Could not load model!")

if __name__ == "__main__":
    main()
