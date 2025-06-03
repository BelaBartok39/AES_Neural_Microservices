"""
Galois Field GF(2^8) Multiplication Training

This module trains specialized neural networks to learn perfect GF(2^8) multiplication
by specific constants (0x02 and 0x03) used in AES MixColumns operation.

Key insights:
- GF(2^8) multiplication is a deterministic mathematical operation
- We can achieve perfect accuracy by learning the lookup table
- Use these as building blocks for MixColumns decomposition

Usage:
    python train_galois_field_models.py
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
import matplotlib.pyplot as plt

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

def galois_multiply(a, b):
    """
    Multiply two numbers in GF(2^8) using AES irreducible polynomial
    
    Args:
        a, b: Numbers to multiply (0-255)
        
    Returns:
        Product in GF(2^8) (0-255)
    """
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        high_bit_set = a & 0x80
        a <<= 1
        if high_bit_set:
            a ^= 0x1B  # AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return p & 0xFF

def generate_gf_multiplication_table(multiplier):
    """
    Generate complete GF(2^8) multiplication table for a specific multiplier
    
    Args:
        multiplier: The constant to multiply by (e.g., 0x02, 0x03)
        
    Returns:
        lookup_table: Array where lookup_table[i] = i * multiplier in GF(2^8)
    """
    lookup_table = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        lookup_table[i] = galois_multiply(i, multiplier)
    
    return lookup_table

def generate_gf_dataset(multiplier, num_samples, approach='classification'):
    """
    Generate dataset for learning GF(2^8) multiplication by a specific constant
    
    Args:
        multiplier: The constant to multiply by (0x02 or 0x03)
        num_samples: Number of training samples
        approach: 'classification' (one-hot) or 'regression' (normalized)
        
    Returns:
        Dictionary with training data
    """
    print(f"Generating GF(2^8) × 0x{multiplier:02x} dataset with {num_samples} samples...")
    
    # Generate the complete lookup table
    lookup_table = generate_gf_multiplication_table(multiplier)
    
    # Print some examples
    print(f"GF(2^8) × 0x{multiplier:02x} examples:")
    for i in [0, 1, 2, 0x53, 0xCA, 0xFF]:
        result = lookup_table[i]
        print(f"  0x{i:02x} × 0x{multiplier:02x} = 0x{result:02x}")
    
    # Generate random input values (we could also use all 256 values multiple times)
    if num_samples <= 256:
        # Use each value at least once
        X = np.arange(256, dtype=np.uint8)[:num_samples]
    else:
        # Random sampling with repetition
        X = np.random.randint(0, 256, size=num_samples, dtype=np.uint8)
    
    # Get corresponding outputs
    y = lookup_table[X]
    
    # Prepare data based on approach
    if approach == 'classification':
        # One-hot encoding (like successful S-box approach)
        X_onehot = keras.utils.to_categorical(X, num_classes=256)
        y_onehot = keras.utils.to_categorical(y, num_classes=256)
        
        X_processed = X_onehot.reshape(num_samples, 256).astype(np.float32)
        y_processed = y_onehot.reshape(num_samples, 256).astype(np.float32)
        
        print(f"Using classification approach (one-hot encoding)")
        
    elif approach == 'regression':
        # Normalized values
        X_processed = X.astype(np.float32) / 255.0
        y_processed = y.astype(np.float32) / 255.0
        
        X_processed = X_processed.reshape(num_samples, 1)
        y_processed = y_processed.reshape(num_samples, 1)
        
        print(f"Using regression approach (normalized values)")
        
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    print(f"Dataset shapes: X={X_processed.shape}, y={y_processed.shape}")
    
    return {
        'X': X_processed,
        'y': y_processed, 
        'X_raw': X,
        'y_raw': y,
        'lookup_table': lookup_table,
        'multiplier': multiplier,
        'approach': approach
    }

def create_gf_classification_model():
    """
    Create model for GF multiplication using classification approach
    (Similar to successful S-box model)
    """
    print("Creating GF multiplication classification model...")
    
    model = keras.Sequential([
        layers.Input(shape=(256,)),  # One-hot input
        
        # Architecture similar to successful S-box model
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='softmax')  # One-hot output
    ])
    
    return model

def create_gf_regression_model():
    """
    Create model for GF multiplication using regression approach
    """
    print("Creating GF multiplication regression model...")
    
    model = keras.Sequential([
        layers.Input(shape=(1,)),  # Single normalized input
        
        # More complex architecture for learning the mapping
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        
        layers.Dense(1, activation='sigmoid')  # Single normalized output
    ])
    
    return model

def train_gf_model(multiplier, approach='classification', num_samples=50000):
    """
    Train a GF(2^8) multiplication model for a specific multiplier
    
    Args:
        multiplier: The constant to multiply by (0x02 or 0x03)
        approach: 'classification' or 'regression'
        num_samples: Number of training samples
        
    Returns:
        Trained model and training history
    """
    print("="*80)
    print(f"TRAINING GF(2^8) × 0x{multiplier:02x} MODEL - {approach.upper()} APPROACH")
    print("="*80)
    
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 100
    PATIENCE = 15
    
    # Generate dataset
    dataset = generate_gf_dataset(multiplier, num_samples, approach)
    
    X, y = dataset['X'], dataset['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    if approach == 'classification':
        model = create_gf_classification_model()
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        model = create_gf_regression_model()
        loss = 'mse'
        metrics = ['mae']
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
        loss=loss,
        metrics=metrics
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f'models/gf_mul{multiplier:02x}_{approach}_checkpoint.keras',
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
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test metrics: {test_metrics}")
    
    # Test perfect accuracy
    perfect_accuracy = test_perfect_gf_multiplication(model, dataset)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/gf_mul{multiplier:02x}_{approach}_model.keras'
    model.save(model_path)
    
    # Save metadata
    metadata = {
        'model_name': f'GF(2^8) × 0x{multiplier:02x} {approach.capitalize()} Model',
        'training_date': datetime.now().isoformat(),
        'approach': approach,
        'multiplier': f'0x{multiplier:02x}',
        'multiplier_int': int(multiplier),
        'perfect_accuracy': float(perfect_accuracy),
        'test_metrics': test_metrics,
        'samples': num_samples,
        'training_time': training_time,
        'lookup_table': dataset['lookup_table'].tolist()
    }
    
    with open(f'models/gf_mul{multiplier:02x}_{approach}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Metadata saved with perfect accuracy: {perfect_accuracy:.4f}")
    
    return model, history, perfect_accuracy

def test_perfect_gf_multiplication(model, dataset):
    """
    Test if the model learned perfect GF(2^8) multiplication
    
    Args:
        model: Trained model
        dataset: Dataset dictionary
        
    Returns:
        Perfect accuracy score (0.0 to 1.0)
    """
    print("\nTesting perfect GF multiplication learning...")
    
    approach = dataset['approach']
    lookup_table = dataset['lookup_table']
    multiplier = dataset['multiplier']
    
    # Test on all possible inputs (0-255)
    all_inputs = np.arange(256, dtype=np.uint8)
    expected_outputs = lookup_table[all_inputs]
    
    # Prepare inputs based on approach
    if approach == 'classification':
        X_test = keras.utils.to_categorical(all_inputs, num_classes=256)
        X_test = X_test.reshape(256, 256).astype(np.float32)
    else:
        X_test = all_inputs.astype(np.float32) / 255.0
        X_test = X_test.reshape(256, 1)
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # Convert predictions back to integers
    if approach == 'classification':
        predicted_outputs = np.argmax(predictions, axis=1)
    else:
        predicted_outputs = np.round(predictions.flatten() * 255).astype(np.uint8)
    
    # Calculate accuracy
    exact_matches = np.sum(predicted_outputs == expected_outputs)
    perfect_accuracy = exact_matches / 256
    
    print(f"Perfect accuracy: {perfect_accuracy:.4f} ({exact_matches}/256)")
    
    # Show some examples
    print(f"\nSample GF(2^8) × 0x{multiplier:02x} predictions:")
    test_values = [0, 1, 2, 0x53, 0xCA, 0xFF]
    for val in test_values:
        expected = expected_outputs[val]
        predicted = predicted_outputs[val]
        correct = "✅" if predicted == expected else "❌"
        print(f"  0x{val:02x} × 0x{multiplier:02x} = 0x{expected:02x} → 0x{predicted:02x} {correct}")
    
    # Analyze errors if any
    if exact_matches < 256:
        errors = np.where(predicted_outputs != expected_outputs)[0]
        print(f"\nErrors found at inputs: {errors[:10]}..." if len(errors) > 10 else f"\nErrors found at inputs: {errors}")
        
        # Show first few errors
        for i in errors[:3]:
            expected = expected_outputs[i]
            predicted = predicted_outputs[i]
            print(f"  Error: 0x{i:02x} × 0x{multiplier:02x} = 0x{expected:02x}, predicted 0x{predicted:02x}")
    
    return perfect_accuracy

def analyze_gf_properties():
    """Analyze mathematical properties of GF(2^8) multiplication"""
    print("\n" + "="*60)
    print("ANALYZING GF(2^8) MULTIPLICATION PROPERTIES")
    print("="*60)
    
    multipliers = [0x02, 0x03]
    
    for mult in multipliers:
        print(f"\nGF(2^8) × 0x{mult:02x} analysis:")
        
        lookup_table = generate_gf_multiplication_table(mult)
        
        # Analyze distribution
        unique_outputs = len(np.unique(lookup_table))
        print(f"  Unique outputs: {unique_outputs}/256")
        
        # Check if it's a bijection (should be for non-zero multipliers)
        if unique_outputs == 256:
            print("  ✅ Bijective mapping (invertible)")
        else:
            print("  ❌ Not bijective")
        
        # Analyze bit patterns
        input_bits = np.unpackbits(np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1)
        output_bits = np.unpackbits(lookup_table.reshape(-1, 1), axis=1)
        
        # Calculate correlation between input and output bits
        print("  Bit correlations (input bit → output bit):")
        for out_bit in range(8):
            correlations = []
            for in_bit in range(8):
                corr = np.corrcoef(input_bits[:, in_bit], output_bits[:, out_bit])[0, 1]
                correlations.append(corr)
            max_corr = max(correlations, key=abs)
            print(f"    Out bit {out_bit}: max correlation = {max_corr:.3f}")
        
        # Check linearity properties
        print("  Checking XOR linearity...")
        linear_violations = 0
        for a in range(0, 256, 17):  # Sample to avoid too many checks
            for b in range(0, 256, 19):
                # Check if mult(a XOR b) = mult(a) XOR mult(b)
                expected = lookup_table[a] ^ lookup_table[b]
                actual = lookup_table[a ^ b]
                if expected != actual:
                    linear_violations += 1
        
        if linear_violations == 0:
            print("  ✅ XOR linear (multiplication distributes over XOR)")
        else:
            print(f"  ❌ Not XOR linear ({linear_violations} violations found)")

def train_all_gf_models():
    """Train all required GF multiplication models"""
    print("="*100)
    print("TRAINING ALL GALOIS FIELD MULTIPLICATION MODELS")
    print("Needed for AES MixColumns decomposition")
    print("="*100)
    
    # Analyze GF properties first
    analyze_gf_properties()
    
    # Train models for both multipliers used in MixColumns
    multipliers = [0x02, 0x03]
    approaches = ['classification', 'regression']
    
    results = {}
    
    for multiplier in multipliers:
        results[f'0x{multiplier:02x}'] = {}
        
        for approach in approaches:
            print(f"\n{'='*60}")
            print(f"Training GF(2^8) × 0x{multiplier:02x} with {approach} approach")
            print(f"{'='*60}")
            
            model, history, accuracy = train_gf_model(multiplier, approach, num_samples=50000)
            
            results[f'0x{multiplier:02x}'][approach] = {
                'model': model,
                'accuracy': accuracy,
                'history': history
            }
    
    # Summary and selection of best models
    print("\n" + "="*100)
    print("GALOIS FIELD MODEL TRAINING SUMMARY")
    print("="*100)
    
    print(f"{'Multiplier':<12} {'Approach':<15} {'Perfect Accuracy':<18}")
    print("-" * 50)
    
    best_models = {}
    
    for mult_hex, mult_results in results.items():
        best_accuracy = 0
        best_approach = None
        
        for approach, result in mult_results.items():
            accuracy = result['accuracy']
            print(f"{mult_hex:<12} {approach:<15} {accuracy:<18.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_approach = approach
        
        best_models[mult_hex] = {
            'approach': best_approach,
            'accuracy': best_accuracy
        }
        
        print(f"  → Best: {best_approach} ({best_accuracy:.4f})")
        print()
    
    # Save the best models as primary GF models
    print("Saving best models as primary GF multiplication models...")
    
    for mult_hex, best_info in best_models.items():
        mult_int = int(mult_hex, 16)
        approach = best_info['approach']
        
        # Load and save the best model
        src_path = f'models/gf_mul{mult_int:02x}_{approach}_model.keras'
        dst_path = f'models/gf_mul{mult_int:02x}_model.keras'
        
        model = keras.models.load_model(src_path)
        model.save(dst_path)
        
        # Update metadata
        with open(f'models/gf_mul{mult_int:02x}_{approach}_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        metadata['selected_approach'] = approach
        metadata['selection_reason'] = 'highest_perfect_accuracy'
        
        with open(f'models/gf_mul{mult_int:02x}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ {mult_hex}: {dst_path} (accuracy: {best_info['accuracy']:.4f})")
    
    print(f"\n🎉 GALOIS FIELD TRAINING COMPLETE!")
    print("✅ GF(2^8) × 0x02 model ready")
    print("✅ GF(2^8) × 0x03 model ready")
    print("🚀 Ready to build MixColumns microservice!")
    
    return results

def main():
    """Main function"""
    print("Galois Field GF(2^8) Multiplication Neural Network Training")
    print("Building blocks for AES MixColumns operation")
    print("="*80)
    
    # Setup
    gpu_available = setup_gpu()
    os.makedirs('models', exist_ok=True)
    
    # Train all required models
    results = train_all_gf_models()
    
    print("\n🎯 Galois Field training complete!")
    print("Ready to build MixColumns microservice using these GF building blocks.")

if __name__ == "__main__":
    main()