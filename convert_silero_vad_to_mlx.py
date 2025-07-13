#!/usr/bin/env python3
"""
Convert Silero VAD PyTorch model to MLX format
"""

import json
import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import argparse
from typing import Dict, Any

class SileroVADConverter:
    """Converter for Silero VAD from PyTorch to MLX"""
    
    def __init__(self):
        self.model_config = {
            "sample_rate": 16000,
            "window_size_samples": 512,
            "input_size": 1,  # Will be determined from model
            "hidden_size": 64,  # Will be determined from model
            "num_classes": 1,
            "model_type": "lstm"
        }
    
    def download_silero_vad(self):
        """Download Silero VAD model from torch.hub"""
        print("Downloading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        print("✓ Model downloaded")
        return model, utils
    
    def extract_model_architecture(self, torch_model):
        """Extract architecture details from PyTorch model"""
        print("\nAnalyzing model architecture...")
        
        # Get model state dict
        state_dict = torch_model.state_dict()
        
        # Analyze layers
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")
        
        # Try to determine LSTM parameters
        if '_model.lstm.weight_ih_l0' in state_dict:
            # Input size from weight_ih_l0
            weight_ih = state_dict['_model.lstm.weight_ih_l0']
            self.model_config['hidden_size'] = weight_ih.shape[0] // 4  # LSTM has 4 gates
            self.model_config['input_size'] = weight_ih.shape[1]
            print(f"\n✓ Detected architecture:")
            print(f"  Input size: {self.model_config['input_size']}")
            print(f"  Hidden size: {self.model_config['hidden_size']}")
        
        # Check for output layer
        if '_model.fc.weight' in state_dict:
            fc_weight = state_dict['_model.fc.weight']
            self.model_config['num_classes'] = fc_weight.shape[0]
            print(f"  Output size: {self.model_config['num_classes']}")
        
        return state_dict
    
    def convert_lstm_weights(self, torch_state_dict) -> Dict[str, mx.array]:
        """Convert PyTorch LSTM weights to MLX format"""
        print("\nConverting LSTM weights...")
        mlx_weights = {}
        
        # Map PyTorch LSTM naming to MLX
        lstm_mappings = {
            '_model.lstm.weight_ih_l0': 'lstm1.Wih',
            '_model.lstm.weight_hh_l0': 'lstm1.Whh', 
            '_model.lstm.bias_ih_l0': 'lstm1.bih',
            '_model.lstm.bias_hh_l0': 'lstm1.bhh',
            '_model.lstm.weight_ih_l1': 'lstm2.Wih',
            '_model.lstm.weight_hh_l1': 'lstm2.Whh',
            '_model.lstm.bias_ih_l1': 'lstm2.bih',
            '_model.lstm.bias_hh_l1': 'lstm2.bhh',
        }
        
        # Check if model has stacked LSTMs
        has_lstm2 = '_model.lstm.weight_ih_l1' in torch_state_dict
        
        if not has_lstm2:
            # Single LSTM layer
            print("  Single LSTM layer detected")
            lstm_mappings = {
                '_model.lstm.weight_ih_l0': 'lstm1.Wih',
                '_model.lstm.weight_hh_l0': 'lstm1.Whh',
                '_model.lstm.bias_ih_l0': 'lstm1.bih',
                '_model.lstm.bias_hh_l0': 'lstm1.bhh',
            }
        else:
            print("  Two LSTM layers detected")
        
        # Convert LSTM weights
        for torch_name, mlx_name in lstm_mappings.items():
            if torch_name in torch_state_dict:
                weight = torch_state_dict[torch_name].numpy()
                mlx_weights[mlx_name] = mx.array(weight)
                print(f"  ✓ Converted {torch_name} -> {mlx_name}: {weight.shape}")
        
        # Convert output layer
        if '_model.fc.weight' in torch_state_dict:
            weight = torch_state_dict['_model.fc.weight'].numpy()
            mlx_weights['output.weight'] = mx.array(weight)
            print(f"  ✓ Converted fc.weight -> output.weight: {weight.shape}")
        
        if '_model.fc.bias' in torch_state_dict:
            bias = torch_state_dict['_model.fc.bias'].numpy()
            mlx_weights['output.bias'] = mx.array(bias)
            print(f"  ✓ Converted fc.bias -> output.bias: {bias.shape}")
        
        return mlx_weights
    
    def create_mlx_model_code(self, has_two_lstms: bool) -> str:
        """Generate MLX model code"""
        if has_two_lstms:
            return '''import mlx.core as mx
import mlx.nn as nn

class SileroVADMLX(nn.Module):
    """MLX implementation of Silero VAD"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(
            config["hidden_size"],
            config["num_classes"]
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def __call__(self, x):
        # First LSTM
        x, _ = self.lstm1(x)
        # Second LSTM  
        x, _ = self.lstm2(x)
        # Output projection
        x = self.output(x)
        # Apply sigmoid
        x = self.sigmoid(x)
        return x
'''
        else:
            return '''import mlx.core as mx
import mlx.nn as nn

class SileroVADMLX(nn.Module):
    """MLX implementation of Silero VAD"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Single LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(
            config["hidden_size"],
            config["num_classes"]
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def __call__(self, x):
        # LSTM
        x, _ = self.lstm1(x)
        # Take last timestep output
        if len(x.shape) == 3:
            x = x[:, -1, :]  # (batch, hidden)
        # Output projection
        x = self.output(x)
        # Apply sigmoid
        x = self.sigmoid(x)
        return x
'''
    
    def save_mlx_model(self, weights: Dict[str, mx.array], output_dir: Path):
        """Save MLX model and config"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        weights_dict = {k: np.array(v) for k, v in weights.items()}
        np.savez(output_dir / "weights.npz", **weights_dict)
        print(f"\n✓ Saved weights to {output_dir / 'weights.npz'}")
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.model_config, f, indent=2)
        print(f"✓ Saved config to {output_dir / 'config.json'}")
        
        # Determine if we have two LSTMs
        has_two_lstms = any('lstm2' in k for k in weights.keys())
        
        # Save model code
        model_code = self.create_mlx_model_code(has_two_lstms)
        with open(output_dir / "model.py", "w") as f:
            f.write(model_code)
        print(f"✓ Saved model code to {output_dir / 'model.py'}")
    
    def test_converted_model(self, output_dir: Path):
        """Test the converted model"""
        print(f"\nTesting converted model...")
        
        # Load config
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        
        # Create dummy input
        batch_size = 1
        seq_length = 32  # 512 samples / 16 window size
        input_size = config["input_size"]
        
        dummy_input = mx.random.normal((batch_size, seq_length, input_size))
        print(f"  Input shape: {dummy_input.shape}")
        
        # Try to run inference
        try:
            # Import the model
            import sys
            sys.path.insert(0, str(output_dir))
            from model import SileroVADMLX
            
            # Create model
            model = SileroVADMLX(config)
            
            # Load weights
            weights = mx.load(str(output_dir / "weights.npz"))
            model.load_weights(list(weights.items()))
            
            # Run inference
            output = model(dummy_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{float(output.min()):.4f}, {float(output.max()):.4f}]")
            print("✓ Model test passed!")
            
        except Exception as e:
            print(f"✗ Model test failed: {e}")
    
    def convert(self, output_dir: str = "mlx_models/vad/silero_vad_mlx"):
        """Main conversion function"""
        print("Silero VAD to MLX Converter")
        print("=" * 50)
        
        # Download model
        torch_model, utils = self.download_silero_vad()
        
        # Extract architecture
        state_dict = self.extract_model_architecture(torch_model)
        
        # Convert weights
        mlx_weights = self.convert_lstm_weights(state_dict)
        
        # Save model
        output_path = Path(output_dir)
        self.save_mlx_model(mlx_weights, output_path)
        
        # Test model
        self.test_converted_model(output_path)
        
        print(f"\n{'='*50}")
        print(f"✓ Conversion complete!")
        print(f"Model saved to: {output_path}")
        print(f"\nTo use the model:")
        print(f"  from whisperx.vad_mlx import load_vad_model_mlx")
        print(f'  vad = load_vad_model_mlx("{output_path}")')

def main():
    parser = argparse.ArgumentParser(description="Convert Silero VAD to MLX format")
    parser.add_argument(
        "--output",
        default="mlx_models/vad/silero_vad_mlx",
        help="Output directory for MLX model"
    )
    args = parser.parse_args()
    
    converter = SileroVADConverter()
    converter.convert(args.output)

if __name__ == "__main__":
    main()