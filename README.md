## Data
Put huggingface data in `./MIR` and unzip `./MIR/images.zip`.

## Inference
```bash
python I2T_inference.py --engine phi3-vision --dataset codeu
```
Results will be saved in `results` folder.

## Evaluation
```bash
python I2T_evaluate.py --engine phi3-vision --dataset codeu

```