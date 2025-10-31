# EE 641 - Homework 3: Attention Mechanisms and Transformers
- **Name**: Yu Hung Kung
- **USCID**: 3431428440
- **Email**: yuhungku@usc.edu

Starter code for implementing multi-head attention and analyzing positional encoding strategies.

## Requirements

```bash
pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
```

## Quick Start

### Problem 1: Multi-Head Attention
```bash
cd problem1
python generate_data.py --seed 641
python train.py
python analyze.py --model-path results/best_model.pth
```

### Problem 2: Positional Encoding
```bash
cd problem2
python generate_data.py --seed 641
python train.py --encoding sinusoidal
python train.py --encoding learned
python train.py --encoding none
python analyze.py

# modified parameters
python train.py --encoding sinusoidal --epochs 50 --lr 5e-4 --batch-size 32
python train.py --encoding learned --epochs 50 --lr 1e-3 --batch-size 32
python train.py --encoding none --epochs 50 --lr 5e-4 --batch-size 32
```

## Full Assignment

See the course website for complete assignment instructions, deliverables, and submission requirements.
