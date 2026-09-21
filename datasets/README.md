# Datasets Directory

This directory stores real or generated synthetic audio datasets used for model training.

## How to populate:
1. Place `.wav` files into two subfolders:
   - `datasets/real/` (Real human voice samples)
   - `datasets/synthetic/` (AI generated / cloned voice samples)
2. Run training script:
   ```bash
   python ml/train.py
   ```
3. Or generate a benchmark synthetic training dataset automatically:
   ```bash
   python ml/train.py --generate-synthetic
   ```
