"""
ShiftRows Training using Permutation Learning Approach

This trains a neural network to learn the ShiftRows transformation by treating it
as a fixed permutation operation. The network learns to map input byte positions
to output byte positions.

Key insights:
- ShiftRows is a deterministic permutation (no key involved)
- Each byte has a fixed destination position
- We can achieve perfect accuracy by learning this mapping

Usage:
    python train_shiftrows_model.py
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

def bytes_to_state(data):
    """Convert 16 bytes to 4x4 AES state matrix"""
    state = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            state[j, i] = data[i * 4 + j]
    return state

def state_to_bytes(state):
    """Convert 4x4 AES state matrix to 16 bytes"""
    data = np.zeros(16, dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            data[i * 4 + j] = state[j, i]
    return data

def shift_rows(state):
    """Apply ShiftRows transformation to state"""
    # Make a copy to avoid modifying the original
    shifted = state.copy()
    
    # Shift each row
    shifted[1] = np.roll(state[1], -1)  # Row 1: shift left by 1
    shifted[2] = np.roll(state[2], -2)  # Row 2: shift left by 2
    shifted[3] = np.roll(state[3], -3)  # Row 3: shift left by 3
    
    return shifted

def get_shiftrows_permutation():
    """
    Get the permutation indices for ShiftRows
    Returns array where output[i] = input[perm[i]]
    """
    # Create identity state with position indices
    identity = np.arange(16, dtype=np.uint8)
    identity_state = bytes_to_state(identity)
    
    # Apply ShiftRows
    shifted_state = shift_rows(identity_state)
    
    # Convert back to get permutation
    permutation = state_to_bytes(shifted_state)
    
    return permutation

def generate_shiftrows_dataset(num_samples):
    """
    Generate dataset for learning the ShiftRows transformation
    
    We'll use multiple approaches:
    1. Direct byte mapping (normalized)
    2. Binary representation for fine-grained learning
    """
    print(f"Generating ShiftRows dataset with {num_samples} samples...")
    
    # Get the permutation pattern
    perm = get_shiftrows_permutation()
    print(f"ShiftRows permutation: {perm}")
    
    # Generate random input blocks
    X = np.random.randint(0, 256, size=(num_samples, 16), dtype=np.uint8)
    y = np.zeros_like(X)
    
    # Apply ShiftRows to each block
    for i in range(num_samples):
        input_state = bytes_to_state(X[i])
        shifted_state = shift_rows(input_state)
        y[i] = state_to_bytes(shifted_state)
    
    # Verify with permutation
    # y[i] should equal X[i][perm]
    for i in range(min(5, num_samples)):  # Check first 5
        expected = X[i][perm]
        assert np.array_equal(y[i], expected), f"Permutation check failed at sample {i}"
    
    print("✅ Permutation verification passed")
    
    # Convert to different representations for training
    
    # Approach 1: Normalized byte values [0, 1]
    X_norm = X.astype(np.float32) / 255.0
    y_norm = y.astype(np.float32) / 255.0
    
    # Approach 2: Binary representation (for compatibility with piecewise trainer)
    X_bin = np.unpackbits(X, axis=1).astype(np.float32)
    y_bin = np.unpackbits(y, axis=1).astype(np.float32)
    
    print(f"Dataset shapes:")
    print(f"  Normalized: X={X_norm.shape}, y={y_norm.shape}")
    print(f"  Binary: X={X_bin.shape}, y={y_bin.shape}")
    
    return {
        'X_norm': X_norm, 'y_norm': y_norm,
        'X_bin': X_bin, 'y_bin': y_bin,
        'X_raw': X, 'y_raw': y,
        'permutation': perm
    }

def create_permutation_model(input_shape, output_shape):
    """
    Create a model optimized for learning permutations
    
    This architecture is designed to learn the fixed mapping efficiently
    """
    print("Creating permutation learning model...")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer with more units to capture position relationships
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(output_shape, activation='sigmoid')
    ])
    
    return model

def create_position_aware_model(input_shape, output_shape):
    """Create a position-aware model that explicitly models byte positions"""
    print("Creating position-aware model...")
    inputs = layers.Input(shape=input_shape)
    
    # Main input processing
    input_reshaped = layers.Reshape((input_shape[0], 1))(inputs)  # Shape: (batch, 16, 1)

    # Position encoding
    positions = tf.range(start=0, limit=input_shape[0], dtype=tf.float32)
    positions = positions / input_shape[0]  # Normalize
    pos_encoding = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(positions, 0), [tf.shape(x)[0], 1])
    )(inputs)
    pos_encoding_reshaped = layers.Reshape((input_shape[0], 1))(pos_encoding)  # Shape: (batch, 16, 1)

    # Concatenate position info with input
    concat = layers.Concatenate(axis=-1)([input_reshaped, pos_encoding_reshaped])  # Shape: (batch, 16, 2)

    # Transformer block
    x = layers.Dense(256)(concat)
    x = layers.BatchNormalization()(x)
    
    # Multi-head attention with explicit axis specification
    attn_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        attention_axes=(1,)  # Attend along sequence dimension
    )(x, x)
    
    # Residual connection
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    # Feed-forward network
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Final processing
    x = layers.Flatten()(x)  # Convert to 2D for output
    outputs = layers.Dense(16, activation='linear')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)


def train_shiftrows_model(approach='binary'):
    """
    Train ShiftRows model using the specified approach
    
    Args:
        approach: 'binary' or 'normalized' or 'position_aware'
    """
    print("="*80)
    print(f"TRAINING SHIFTROWS MODEL - {approach.upper()} APPROACH")
    print("="*80)
    
    # Configuration
    NUM_SAMPLES = 100000  # Less than S-box since this is simpler
    BATCH_SIZE = 256
    EPOCHS = 50
    PATIENCE = 10
    
    # Generate dataset
    dataset = generate_shiftrows_dataset(NUM_SAMPLES)
    
    # Select data based on approach
    if approach == 'binary':
        X, y = dataset['X_bin'], dataset['y_bin']
        model = create_permutation_model(X.shape[1:], y.shape[1])
    elif approach == 'normalized':
        X, y = dataset['X_norm'], dataset['y_norm']
        model = create_permutation_model(X.shape[1:], y.shape[1])
    elif approach == 'position_aware':
        X, y = dataset['X_norm'], dataset['y_norm']
        model = create_position_aware_model(X.shape[1:], y.shape[1])
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse' if approach == 'normalized' else 'binary_crossentropy',
        metrics=['mae'] if approach == 'normalized' else ['accuracy']
    )
    
    # Model summary
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
            filepath=f'models/shiftrows_{approach}_checkpoint.keras',
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
    
    # Test on specific examples
    print("\nTesting on specific patterns...")
    test_perfect_shiftrows(model, dataset, approach)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/shiftrows_{approach}_model.keras'
    model.save(model_path)
    
    # Save metadata
    metadata = {
        'model_name': f'ShiftRows {approach.capitalize()} Model',
        'training_date': datetime.now().isoformat(),
        'approach': approach,
        'accuracy': float(test_metrics[1]) if approach == 'binary' else float(1.0 - test_metrics[1]),
        'samples': NUM_SAMPLES,
        'training_time': training_time,
        'permutation': dataset['permutation'].tolist()
    }
    
    with open(f'models/shiftrows_{approach}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, history, test_metrics

def test_perfect_shiftrows(model, dataset, approach):
    """Test if the model learned perfect ShiftRows"""
    print("\nTesting perfect ShiftRows learning...")
    
    # Get permutation
    perm = dataset['permutation']
    
    # Test on fresh random data
    test_inputs = np.random.randint(0, 256, size=(100, 16), dtype=np.uint8)
    
    # Prepare inputs based on approach
    if approach == 'binary':
        X_test = np.unpackbits(test_inputs, axis=1).astype(np.float32)
    else:
        X_test = test_inputs.astype(np.float32) / 255.0
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert predictions back to bytes
    if approach == 'binary':
        y_pred_binary = (y_pred > 0.5).astype(np.uint8)
        y_pred_bytes = np.packbits(y_pred_binary, axis=1)
    else:
        y_pred_bytes = np.round(y_pred * 255).astype(np.uint8)
    
    # Calculate expected outputs
    expected_outputs = np.zeros_like(test_inputs)
    for i in range(len(test_inputs)):
        expected_outputs[i] = test_inputs[i][perm]
    
    # Check accuracy
    exact_matches = np.sum(np.all(y_pred_bytes == expected_outputs, axis=1))
    byte_matches = np.sum(y_pred_bytes == expected_outputs)
    total_bytes = len(test_inputs) * 16
    
    exact_accuracy = exact_matches / len(test_inputs)
    byte_accuracy = byte_matches / total_bytes
    
    print(f"Exact accuracy: {exact_accuracy:.4f} ({exact_matches}/{len(test_inputs)})")
    print(f"Byte accuracy: {byte_accuracy:.4f} ({byte_matches}/{total_bytes})")
    
    # Show a few examples
    print("\nExample predictions:")
    for i in range(min(3, len(test_inputs))):
        input_hex = ' '.join(f'{b:02x}' for b in test_inputs[i])
        expected_hex = ' '.join(f'{b:02x}' for b in expected_outputs[i])
        pred_hex = ' '.join(f'{b:02x}' for b in y_pred_bytes[i])
        correct = "✅" if np.array_equal(y_pred_bytes[i], expected_outputs[i]) else "❌"
        
        print(f"\nExample {i+1}:")
        print(f"  Input:    {input_hex}")
        print(f"  Expected: {expected_hex}")
        print(f"  Predicted: {pred_hex}")
        print(f"  Status: {correct}")
    
    return exact_accuracy, byte_accuracy

def compare_approaches():
    """Train and compare different approaches"""
    print("="*100)
    print("COMPARING DIFFERENT APPROACHES FOR SHIFTROWS")
    print("="*100)
    
    approaches = ['binary', 'normalized', 'position_aware']
    results = {}
    
    for approach in approaches:
        print(f"\n{'='*50}")
        print(f"Training {approach} approach...")
        print(f"{'='*50}")
        
        model, history, test_metrics = train_shiftrows_model(approach)
        
        # Test perfect accuracy
        dataset = generate_shiftrows_dataset(1000)  # Small dataset for testing
        exact_acc, byte_acc = test_perfect_shiftrows(model, dataset, approach)
        
        results[approach] = {
            'model': model,
            'test_metrics': test_metrics,
            'exact_accuracy': exact_acc,
            'byte_accuracy': byte_acc
        }
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY OF RESULTS")
    print("="*100)
    
    print(f"{'Approach':<15} {'Exact Acc':<12} {'Byte Acc':<12}")
    print("-" * 40)
    
    best_approach = None
    best_accuracy = 0
    
    for approach, result in results.items():
        exact_acc = result['exact_accuracy']
        byte_acc = result['byte_accuracy']
        print(f"{approach:<15} {exact_acc:<12.4f} {byte_acc:<12.4f}")
        
        if exact_acc > best_accuracy:
            best_accuracy = exact_acc
            best_approach = approach
    
    print(f"\n🏆 Best approach: {best_approach} with {best_accuracy:.4f} exact accuracy")
    
    # Save the best model as the primary ShiftRows model
    if best_approach and best_accuracy > 0.95:
        src_path = f'models/shiftrows_{best_approach}_model.keras'
        dst_path = 'models/shiftrows_model.keras'
        
        # Copy the best model
        model = keras.models.load_model(src_path)
        model.save(dst_path)
        
        # Copy metadata
        with open(f'models/shiftrows_{best_approach}_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        metadata['selected_approach'] = best_approach
        metadata['selection_reason'] = 'highest_exact_accuracy'
        
        with open('models/shiftrows_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Best model saved as: {dst_path}")
    
    return results

def main():
    """Main function"""
    print("ShiftRows Neural Network Training")
    print("="*80)
    
    # Setup
    gpu_available = setup_gpu()
    os.makedirs('models', exist_ok=True)
    
    # Option 1: Train a specific approach
    # model, history, metrics = train_shiftrows_model('binary')
    
    # Option 2: Compare all approaches and select the best
    results = compare_approaches()
    
    print("\n🎯 ShiftRows training complete!")
    print("The model is ready for microservice integration.")

if __name__ == "__main__":
    main()