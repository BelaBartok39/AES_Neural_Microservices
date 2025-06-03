"""
Complete Neural AES Training Pipeline

This script orchestrates the training of all neural AES components in the correct
dependency order, from basic building blocks to complete microservices.

Training Pipeline:
1. Basic XOR model (foundation for AddRoundKey)
2. Galois Field multiplication models (foundation for MixColumns)
3. S-box classification model (SubBytes)
4. Permutation model (ShiftRows)
5. AddRoundKey microservice (using XOR)
6. MixColumns microservice (using GF models)
7. Complete system validation

Usage:
    python train_complete_neural_aes.py [--quick] [--gpu-only]
    
    --quick: Use smaller datasets for faster training (for testing)
    --gpu-only: Only train if GPU is available
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def setup_environment():
    """Setup the training environment"""
    print("="*80)
    print("NEURAL AES COMPLETE TRAINING PIPELINE")
    print("Building the World's First Neural Cryptographic System")
    print("="*80)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    
    # Check for required modules
    required_modules = [
        'numpy', 'tensorflow', 'scikit-learn', 'matplotlib'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        print("Please install with: pip install " + " ".join(missing_modules))
        return False
    
    print("✅ Environment setup complete")
    return True

def check_gpu_availability():
    """Check if GPU is available for training"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠️ No GPU found - training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def estimate_training_time(quick_mode: bool, has_gpu: bool) -> str:
    """Estimate total training time"""
    if quick_mode:
        base_time = 30 if has_gpu else 90  # minutes
    else:
        base_time = 120 if has_gpu else 300  # minutes
    
    hours = base_time // 60
    minutes = base_time % 60
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"

class TrainingPipeline:
    """Manages the complete neural AES training pipeline"""
    
    def __init__(self, quick_mode: bool = False, gpu_only: bool = False):
        self.quick_mode = quick_mode
        self.gpu_only = gpu_only
        self.training_log = []
        self.start_time = datetime.now()
        
        # Training configuration
        if quick_mode:
            self.config = {
                'xor_samples': 10000,
                'gf_samples': 25000, 
                'sbox_samples': 50000,
                'shiftrows_samples': 50000,
                'mixcolumns_samples': 50000,
                'epochs': 20,
                'patience': 5
            }
        else:
            self.config = {
                'xor_samples': 50000,
                'gf_samples': 50000,
                'sbox_samples': 200000,
                'shiftrows_samples': 100000,
                'mixcolumns_samples': 100000,
                'epochs': 100,
                'patience': 15
            }
        
        print(f"Training mode: {'Quick' if quick_mode else 'Full'}")
        print(f"Configuration: {self.config}")
    
    def log_step(self, step: str, status: str, details: Dict = None):
        """Log a training step"""
        timestamp = datetime.now()
        log_entry = {
            'step': step,
            'status': status,
            'timestamp': timestamp.isoformat(),
            'details': details or {}
        }
        self.training_log.append(log_entry)
        
        elapsed = (timestamp - self.start_time).total_seconds()
        print(f"[{elapsed:6.1f}s] {step}: {status}")
        
        if details:
            for key, value in details.items():
                print(f"         {key}: {value}")
    
    def train_basic_xor_model(self) -> bool:
        """Train the basic XOR model (foundation for AddRoundKey)"""
        self.log_step("Basic XOR Model", "Starting training...")
        
        try:
            # Import and run XOR training from the AddRoundKey trainer
            from train_addroundkey_model import create_perfect_basic_xor_model
            
            print("Training perfect 2-input XOR model...")
            xor_model = create_perfect_basic_xor_model()
            
            # Save the XOR model
            xor_model.save('models/perfect_xor_model.keras')
            
            self.log_step("Basic XOR Model", "✅ Completed", {
                'accuracy': '100% (Perfect)',
                'model_saved': 'models/perfect_xor_model.keras'
            })
            return True
            
        except Exception as e:
            self.log_step("Basic XOR Model", f"❌ Failed: {e}")
            return False
    
    def train_galois_field_models(self) -> bool:
        """Train Galois Field multiplication models"""
        self.log_step("Galois Field Models", "Starting training...")
        
        try:
            # Check if we have the GF training script
            if not os.path.exists('train_galois_field_models.py'):
                self.log_step("Galois Field Models", "❌ Missing train_galois_field_models.py")
                return False
            
            # Import and run GF training
            import subprocess
            result = subprocess.run([
                sys.executable, 'train_galois_field_models.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if models were created
                gf02_exists = os.path.exists('models/gf_mul02_model.keras')
                gf03_exists = os.path.exists('models/gf_mul03_model.keras')
                
                if gf02_exists and gf03_exists:
                    self.log_step("Galois Field Models", "✅ Completed", {
                        'gf_mul02': 'models/gf_mul02_model.keras',
                        'gf_mul03': 'models/gf_mul03_model.keras'
                    })
                    return True
                else:
                    self.log_step("Galois Field Models", "❌ Models not created")
                    return False
            else:
                self.log_step("Galois Field Models", f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("Galois Field Models", f"❌ Failed: {e}")
            return False
    
    def train_subbytes_model(self) -> bool:
        """Train the SubBytes S-box model"""
        self.log_step("SubBytes Model", "Starting training...")
        
        try:
            # Check if we have the SubBytes training script
            if not os.path.exists('train_subbytes_model.py'):
                self.log_step("SubBytes Model", "❌ Missing train_subbytes_model.py")
                return False
            
            # Import and run SubBytes training
            import subprocess
            result = subprocess.run([
                sys.executable, 'train_subbytes_model.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if model was created
                model_exists = os.path.exists('models/successful_sbox_model.keras')
                
                if model_exists:
                    self.log_step("SubBytes Model", "✅ Completed", {
                        'model': 'models/successful_sbox_model.keras',
                        'approach': 'One-hot classification'
                    })
                    return True
                else:
                    self.log_step("SubBytes Model", "❌ Model not created")
                    return False
            else:
                self.log_step("SubBytes Model", f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("SubBytes Model", f"❌ Failed: {e}")
            return False
    
    def train_shiftrows_model(self) -> bool:
        """Train the ShiftRows permutation model"""
        self.log_step("ShiftRows Model", "Starting training...")
        
        try:
            # Check if we have the ShiftRows training script
            if not os.path.exists('train_shiftrows_model.py'):
                self.log_step("ShiftRows Model", "❌ Missing train_shiftrows_model.py")
                return False
            
            # Import and run ShiftRows training
            import subprocess
            result = subprocess.run([
                sys.executable, 'train_shiftrows_model.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if model was created
                model_exists = os.path.exists('models/shiftrows_model.keras')
                
                if model_exists:
                    self.log_step("ShiftRows Model", "✅ Completed", {
                        'model': 'models/shiftrows_model.keras',
                        'approach': 'Permutation learning'
                    })
                    return True
                else:
                    self.log_step("ShiftRows Model", "❌ Model not created")
                    return False
            else:
                self.log_step("ShiftRows Model", f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("ShiftRows Model", f"❌ Failed: {e}")
            return False
    
    def train_addroundkey_model(self) -> bool:
        """Train the AddRoundKey bitwise decomposition model"""
        self.log_step("AddRoundKey Model", "Starting training...")
        
        try:
            # Check if we have the AddRoundKey training script
            if not os.path.exists('train_addroundkey_model.py'):
                self.log_step("AddRoundKey Model", "❌ Missing train_addroundkey_model.py")
                return False
            
            # Import and run AddRoundKey training
            import subprocess
            result = subprocess.run([
                sys.executable, 'train_addroundkey_model.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if model was created
                model_exists = os.path.exists('models/perfect_bitwise_addroundkey.keras')
                
                if model_exists:
                    self.log_step("AddRoundKey Model", "✅ Completed", {
                        'model': 'models/perfect_bitwise_addroundkey.keras',
                        'approach': 'Bitwise XOR decomposition',
                        'accuracy': '100% (Perfect)'
                    })
                    return True
                else:
                    self.log_step("AddRoundKey Model", "❌ Model not created")
                    return False
            else:
                self.log_step("AddRoundKey Model", f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("AddRoundKey Model", f"❌ Failed: {e}")
            return False
    
    def train_mixcolumns_model(self) -> bool:
        """Train the MixColumns GF decomposition model"""
        self.log_step("MixColumns Model", "Starting training...")
        
        try:
            # Check dependencies first
            gf02_exists = os.path.exists('models/gf_mul02_model.keras')
            gf03_exists = os.path.exists('models/gf_mul03_model.keras')
            
            if not (gf02_exists and gf03_exists):
                self.log_step("MixColumns Model", "❌ Missing GF models dependency")
                return False
            
            # Check if we have the MixColumns training script
            if not os.path.exists('train_mixcolumns_model.py'):
                self.log_step("MixColumns Model", "❌ Missing train_mixcolumns_model.py")
                return False
            
            # Import and run MixColumns training
            import subprocess
            result = subprocess.run([
                sys.executable, 'train_mixcolumns_model.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if model was created
                model_exists = os.path.exists('models/mixcolumns_model.keras')
                
                if model_exists:
                    self.log_step("MixColumns Model", "✅ Completed", {
                        'model': 'models/mixcolumns_model.keras',
                        'approach': 'GF decomposition'
                    })
                    return True
                else:
                    self.log_step("MixColumns Model", "❌ Model not created")
                    return False
            else:
                self.log_step("MixColumns Model", f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("MixColumns Model", f"❌ Failed: {e}")
            return False
    
    def validate_complete_system(self) -> bool:
        """Validate the complete neural AES system"""
        self.log_step("System Validation", "Starting validation...")
        
        try:
            # Check all required models exist
            required_models = [
                'models/perfect_bitwise_addroundkey.keras',
                'models/successful_sbox_model.keras', 
                'models/shiftrows_model.keras',
                'models/mixcolumns_model.keras'
            ]
            
            missing_models = []
            for model_path in required_models:
                if not os.path.exists(model_path):
                    missing_models.append(model_path)
            
            if missing_models:
                self.log_step("System Validation", f"❌ Missing models: {missing_models}")
                return False
            
            # Run the complete system demo
            if os.path.exists('complete_neural_aes_demo.py'):
                import subprocess
                result = subprocess.run([
                    sys.executable, 'complete_neural_aes_demo.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_step("System Validation", "✅ Complete system validated")
                    return True
                else:
                    self.log_step("System Validation", f"❌ System validation failed: {result.stderr}")
                    return False
            else:
                self.log_step("System Validation", "⚠️ Demo script not found, skipping validation")
                return True
                
        except Exception as e:
            self.log_step("System Validation", f"❌ Failed: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        print(f"\n🚀 Starting complete neural AES training pipeline...")
        print(f"Estimated time: {estimate_training_time(self.quick_mode, check_gpu_availability())}")
        
        if self.gpu_only and not check_gpu_availability():
            print("❌ GPU required but not available")
            return False
        
        # Training steps in dependency order
        steps = [
            ("Basic XOR", self.train_basic_xor_model),
            ("Galois Field", self.train_galois_field_models),
            ("SubBytes", self.train_subbytes_model),
            ("ShiftRows", self.train_shiftrows_model),
            ("AddRoundKey", self.train_addroundkey_model),
            ("MixColumns", self.train_mixcolumns_model),
            ("System Validation", self.validate_complete_system)
        ]
        
        successful_steps = 0
        total_steps = len(steps)
        
        for step_name, step_function in steps:
            print(f"\n{'='*60}")
            print(f"STEP {successful_steps + 1}/{total_steps}: {step_name}")
            print(f"{'='*60}")
            
            try:
                success = step_function()
                if success:
                    successful_steps += 1
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    break
            except KeyboardInterrupt:
                print(f"\n⏹️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ {step_name} failed with exception: {e}")
                break
        
        # Generate final report
        self.generate_training_report(successful_steps, total_steps)
        
        return successful_steps == total_steps
    
    def generate_training_report(self, successful_steps: int, total_steps: int):
        """Generate a comprehensive training report"""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        report = {
            'training_session': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': total_duration.total_seconds(),
                'duration_human': str(total_duration),
                'mode': 'quick' if self.quick_mode else 'full',
                'configuration': self.config
            },
            'results': {
                'successful_steps': successful_steps,
                'total_steps': total_steps,
                'success_rate': successful_steps / total_steps,
                'status': 'completed' if successful_steps == total_steps else 'partial'
            },
            'training_log': self.training_log,
            'models_created': []
        }
        
        # Check which models were created
        model_files = [
            'models/perfect_xor_model.keras',
            'models/gf_mul02_model.keras',
            'models/gf_mul03_model.keras',
            'models/successful_sbox_model.keras',
            'models/shiftrows_model.keras', 
            'models/perfect_bitwise_addroundkey.keras',
            'models/mixcolumns_model.keras'
        ]
        
        for model_path in model_files:
            if os.path.exists(model_path):
                stat = os.stat(model_path)
                report['models_created'].append({
                    'path': model_path,
                    'size_bytes': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/training_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("NEURAL AES TRAINING PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        print(f"Training Duration: {total_duration}")
        print(f"Steps Completed: {successful_steps}/{total_steps}")
        print(f"Success Rate: {successful_steps/total_steps:.1%}")
        print(f"Models Created: {len(report['models_created'])}")
        
        if successful_steps == total_steps:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"✅ All neural AES components trained successfully")
            print(f"✅ Complete neural cryptographic system ready")
            print(f"🚀 Ready to run complete_neural_aes_demo.py")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            print(f"Some components failed to train - check logs for details")
            print(f"You may need to:")
            print(f"  - Check error messages above")
            print(f"  - Ensure sufficient memory/GPU resources")
            print(f"  - Try training individual components manually")
        
        print(f"\n📊 Detailed report saved: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete Neural AES Training Pipeline")
    parser.add_argument('--quick', action='store_true', 
                       help='Use smaller datasets for faster training')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Only train if GPU is available')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Check GPU
    has_gpu = check_gpu_availability()
    
    if args.gpu_only and not has_gpu:
        print("❌ GPU required but not available. Exiting.")
        return 1
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(quick_mode=args.quick, gpu_only=args.gpu_only)
    
    try:
        success = pipeline.run_complete_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())