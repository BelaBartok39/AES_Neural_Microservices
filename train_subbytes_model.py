"""
SubBytes Training using the Successful Architecture from Piecewise Trainer

This uses the exact same successful approaches that achieved 100% S-box accuracy:
- One-hot encoding for input and output
- Categorical crossentropy loss  
- Specific MLP architecture with BatchNorm and Dropout
- Proper training callbacks

Usage:
    python train_subbytes_successful.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

def setup_gpu():
    """Setup GPU/CUDA configuration"""
    print("Setting up GPU/CUDA configuration...")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUs Available: {len(gpus)}")
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using float32 precision")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPUs found, using CPU")
        return False

def get_aes_sbox():
    """Get the official AES S-box"""
    sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0xCC, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4,
        0x79, 0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE,
        0x08, 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B,
        0x8A, 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D,
        0x9E, 0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28,
        0xDF, 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB
    ], dtype=np.uint8)
    
    if len(sbox) != 256:
        raise ValueError(f"S-box must have exactly 256 entries, got {len(sbox)}")
    
    return sbox

def generate_sbox_dataset(num_samples):
    """
    Generate dataset for learning the S-box transformation using the SUCCESSFUL approach
    
    Key insight: Use one-hot encoding for both input and output, treat as classification
    """
    print(f"Generating S-box dataset with {num_samples} samples...")
    
    # Get the S-box
    sbox = get_aes_sbox()
    
    # Generate random input bytes (0-255)
    X = np.random.randint(0, 256, size=(num_samples, 1), dtype=np.uint8)
    y = np.zeros_like(X)
    
    # Apply S-box transformation
    for i in range(num_samples):
        y[i, 0] = sbox[X[i, 0]]
    
    # Convert to one-hot encoding for inputs and outputs (KEY INSIGHT!)
    X_onehot = keras.utils.to_categorical(X, num_classes=256)
    y_onehot = keras.utils.to_categorical(y, num_classes=256)
    
    # Reshape to flatten the one-hot vectors
    X_onehot = X_onehot.reshape(num_samples, 256)
    y_onehot = y_onehot.reshape(num_samples, 256)
    
    print(f"Dataset shapes: X={X_onehot.shape}, y={y_onehot.shape}")
    print(f"Input encoding: One-hot (256 classes)")
    print(f"Output encoding: One-hot (256 classes)")
    
    return X_onehot.astype(np.float32), y_onehot.astype(np.float32), X, y

def create_successful_mlp_model(input_shape, output_shape):
    """
    Create the SUCCESSFUL MLP architecture from the piecewise trainer
    
    This is the exact architecture that achieved 100% S-box accuracy
    """
    print("Creating successful MLP architecture...")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(output_shape, activation='softmax')  # Softmax for classification!
    ])
    
    return model

def train_and_evaluate_sbox():
    """Train and evaluate S-box using the successful approach"""
    print("="*80)
    print("TRAINING S-BOX WITH SUCCESSFUL ARCHITECTURE")
    print("Using the exact approach that achieved 100% accuracy")
    print("="*80)
    
    # Configuration from successful trainer
    NUM_SAMPLES = 200000  # Use same sample size as successful trainer
    BATCH_SIZE = 128
    EPOCHS = 100
    PATIENCE = 20
    
    # Generate dataset using successful approach
    X, y, X_raw, y_raw = generate_sbox_dataset(NUM_SAMPLES)
    
    # Split into train/val/test using same approach
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create the successful model architecture
    model = create_successful_mlp_model(X_train.shape[1:], y_train.shape[1])
    
    # Compile with categorical crossentropy (KEY INSIGHT!)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1),
        loss='categorical_crossentropy',  # Use categorical for classification!
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Setup callbacks like in successful trainer
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath='models/best_sbox_model.keras',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions and analyze
    print("\nAnalyzing predictions...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true classes from test set
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate detailed accuracy
    exact_matches = np.sum(y_pred_classes == y_true_classes)
    total_samples = len(y_true_classes)
    detailed_accuracy = exact_matches / total_samples
    
    print(f"Detailed accuracy: {detailed_accuracy:.4f} ({exact_matches}/{total_samples})")
    
    # Test on all possible S-box inputs (0-255)
    print("\nTesting on complete S-box mapping...")
    
    # Generate all inputs
    all_inputs = np.arange(256).reshape(-1, 1)
    all_inputs_onehot = keras.utils.to_categorical(all_inputs, num_classes=256)
    all_inputs_onehot = all_inputs_onehot.reshape(256, 256)
    
    # Get true S-box outputs
    sbox = get_aes_sbox()
    true_outputs = sbox
    
    # Get model predictions
    pred_outputs_onehot = model.predict(all_inputs_onehot)
    pred_outputs = np.argmax(pred_outputs_onehot, axis=1)
    
    # Calculate complete S-box accuracy
    sbox_accuracy = np.mean(pred_outputs == true_outputs)
    sbox_matches = np.sum(pred_outputs == true_outputs)
    
    print(f"Complete S-box accuracy: {sbox_accuracy:.4f} ({sbox_matches}/256)")
    
    # Show some examples
    print("\nSample S-box predictions:")
    for i in [0, 1, 16, 17, 32, 255]:
        input_val = i
        true_val = true_outputs[i]
        pred_val = pred_outputs[i]
        correct = "✅" if pred_val == true_val else "❌"
        print(f"  S-box[0x{input_val:02x}] = 0x{true_val:02x} → 0x{pred_val:02x} {correct}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/successful_sbox_model.keras')
    
    # Save metadata
    metadata = {
        'model_name': 'Successful S-box Model',
        'training_date': datetime.now().isoformat(),
        'approach': 'one_hot_classification',
        'accuracy': float(sbox_accuracy),
        'method': 'One-hot encoding + categorical crossentropy',
        'architecture': 'MLP with BatchNorm and Dropout',
        'samples': NUM_SAMPLES,
        'training_time': training_time,
        'complete_sbox_matches': int(sbox_matches)
    }
    
    with open('models/successful_sbox_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Model saved to: models/successful_sbox_model.keras")
    print(f"💾 Metadata saved to: models/successful_sbox_metadata.json")
    
    # Final assessment
    if sbox_accuracy >= 0.99:
        print(f"\n🎉 EXCELLENT! Achieved {sbox_accuracy:.1%} S-box accuracy!")
        print("This matches the successful trainer performance!")
    elif sbox_accuracy >= 0.90:
        print(f"\n✅ GOOD! Achieved {sbox_accuracy:.1%} S-box accuracy!")
        print("Very close to the successful trainer performance!")
    else:
        print(f"\n⚠️ Achieved {sbox_accuracy:.1%} S-box accuracy")
        print("Still room for improvement to match successful trainer")
    
    return {
        'model': model,
        'history': history.history,
        'test_accuracy': test_acc,
        'sbox_accuracy': sbox_accuracy,
        'training_time': training_time,
        'complete_matches': sbox_matches
    }

class SubBytesProcessor:
    """
    SubBytes processor using the successful trained model
    """
    
    def __init__(self, model):
        self.model = model
        self.sbox = get_aes_sbox()
        print("SubBytesProcessor initialized with successful model")
    
    def process_bytes(self, input_bytes):
        """Process 16-byte SubBytes using the successful model"""
        if len(input_bytes) != 16:
            raise ValueError(f"Expected 16 bytes, got {len(input_bytes)}")
        
        result = np.zeros(16, dtype=np.uint8)
        
        # Process each byte individually through the S-box model
        for i in range(16):
            # Convert byte to one-hot encoding
            input_onehot = keras.utils.to_categorical([input_bytes[i]], num_classes=256)
            input_onehot = input_onehot.reshape(1, 256)
            
            # Get model prediction
            pred_onehot = self.model.predict(input_onehot, verbose=0)
            pred_class = np.argmax(pred_onehot[0])
            
            result[i] = pred_class
        
        return result

def test_subbytes_processor():
    """Test the SubBytes processor with the successful model"""
    print("\n" + "="*70)
    print("TESTING SUBBYTES PROCESSOR WITH SUCCESSFUL MODEL")
    print("="*70)
    
    # Load the trained model
    model_path = 'models/successful_sbox_model.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run training first!")
        return False
    
    # Load model
    model = keras.models.load_model(model_path)
    print(f"✅ Loaded model from {model_path}")
    
    # Create processor
    processor = SubBytesProcessor(model)
    
    # Test cases
    test_cases = [
        {
            'name': 'All zeros',
            'input': np.zeros(16, dtype=np.uint8)
        },
        {
            'name': 'Sequential bytes',
            'input': np.arange(16, dtype=np.uint8)
        },
        {
            'name': 'High values',
            'input': np.arange(240, 256, dtype=np.uint8)
        },
        {
            'name': 'AES test vector state',
            'input': np.array([0x19, 0xa0, 0x9a, 0xe9, 0x3d, 0xf4, 0xc6, 0xf8,
                              0xe3, 0xe2, 0x8d, 0x48, 0xbe, 0x2b, 0x2a, 0x08], dtype=np.uint8)
        },
        {
            'name': 'Random case',
            'input': np.random.randint(0, 256, 16, dtype=np.uint8)
        }
    ]
    
    print(f"\nTesting {len(test_cases)} cases:")
    
    # Get reference S-box
    reference_sbox = get_aes_sbox()
    all_correct = True
    
    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        
        input_bytes = test_case['input']
        
        # True S-box result
        expected = reference_sbox[input_bytes]
        
        # Neural S-box result
        result = processor.process_bytes(input_bytes)
        
        # Check if correct
        exact_match = np.array_equal(result, expected)
        byte_accuracy = np.mean(result == expected)
        
        print(f"Input:     {' '.join(f'{b:02x}' for b in input_bytes)}")
        print(f"Expected:  {' '.join(f'{b:02x}' for b in expected)}")
        print(f"Neural:    {' '.join(f'{b:02x}' for b in result)}")
        print(f"Status:    {'✅ PERFECT' if exact_match else '❌ ERROR'}")
        print(f"Accuracy:  {byte_accuracy:.3f}")
        
        if not exact_match:
            all_correct = False
            wrong_indices = np.where(result != expected)[0]
            print(f"Wrong bytes at positions: {wrong_indices}")
    
    # Benchmark performance
    print(f"\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Generate test data
    n_samples = 1000
    test_data = np.random.randint(0, 256, size=(n_samples, 16), dtype=np.uint8)
    expected_results = reference_sbox[test_data]
    
    print(f"Processing {n_samples} samples...")
    
    # Time the processing
    start_time = time.time()
    
    results = []
    for i in range(n_samples):
        result = processor.process_bytes(test_data[i])
        results.append(result)
    
    results = np.array(results)
    end_time = time.time()
    
    # Calculate metrics
    processing_time = end_time - start_time
    samples_per_second = n_samples / processing_time
    
    exact_matches = np.sum(np.all(results == expected_results, axis=1))
    total_byte_matches = np.sum(results == expected_results)
    total_bytes = n_samples * 16
    
    exact_accuracy = exact_matches / n_samples
    byte_accuracy = total_byte_matches / total_bytes
    
    print(f"Results:")
    print(f"  Exact accuracy: {exact_accuracy:.4f} ({exact_matches}/{n_samples})")
    print(f"  Byte accuracy: {byte_accuracy:.4f} ({total_byte_matches}/{total_bytes})")
    print(f"  Processing time: {processing_time:.3f} seconds")
    print(f"  Throughput: {samples_per_second:.1f} samples/second")
    
    # Final assessment
    print(f"\n" + "="*70)
    if all_correct and exact_accuracy >= 0.99:
        print("🎉 COMPLETE SUCCESS!")
        print("✅ Perfect accuracy on test cases")
        print("✅ High accuracy on benchmark")
        print("✅ Ready for microservice integration!")
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Create SubBytes microservice wrapper")
        print("   2. Integrate with master orchestrator")
        print("   3. Compare with AddRoundKey performance")
        
        return True
    else:
        print("⚠️ NEEDS IMPROVEMENT")
        if not all_correct:
            print("❌ Some test cases failed")
        if exact_accuracy < 0.99:
            print(f"❌ Benchmark accuracy too low: {exact_accuracy:.3f}")
        
        return False

def main():
    """Main function"""
    print("SubBytes Training with Successful Architecture")
    print("Adapting the 100% accuracy approach from piecewise trainer")
    print("="*80)
    
    # Setup
    gpu_available = setup_gpu()
    os.makedirs('models', exist_ok=True)
    
    # Train the S-box model
    try:
        print("\n1. Training S-box model with successful architecture...")
        training_results = train_and_evaluate_sbox()
        
        print("\n2. Testing SubBytes processor...")
        test_success = test_subbytes_processor()
        
        if test_success:
            print("\n🏆 TRAINING SUCCESS!")
            print("Successfully adapted the successful architecture!")
            print("S-box learning achieved using one-hot classification approach!")
        else:
            print("\n📚 Training completed but needs refinement")
            print("The approach is promising but may need parameter tuning")
            
        return training_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    
    if results and results['sbox_accuracy'] >= 0.99:
        print("\n🎯 MISSION ACCOMPLISHED!")
        print("Successfully replicated the successful trainer's approach!")
        print("This proves that the architecture and method are key to success!")
    else:
        print("\n🔄 ITERATION NEEDED")
        print("The approach is on the right track but needs refinement")