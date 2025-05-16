# Multi-Headed Deep Feature Learning for PPGL Genetic Cluster Classification

This repository contains the implementation of a novel multi-headed deep learning architecture for the classification of Pheochromocytoma and Paraganglioma (PPGL) genetic clusters from CT images. The model leverages anatomical location-aware feature learning to improve classification accuracy across different genetic subtypes.

## Overview

Pheochromocytoma and paraganglioma (PPGLs) are rare neuroendocrine tumors that pose significant diagnostic challenges due to their heterogeneous appearance and complex anatomical locations[1]. The treatment and outcome of PPGLs depend on their four genetic subtypes:

- SDHx
- VHL/EPAS1
- Kinase signaling
- Sporadic

This work introduces a novel weighted multi-headed deep learning architecture incorporating anatomical location-specific (head and neck, chest, and abdomen to pelvic) feature extraction to enhance PPGL tumor's genetic type classification.

## Architecture

The model architecture consists of:

1. **Swin Transformer Backbone**: Extracts high-dimensional features from CT images
2. **Anatomical Location Weight Estimator**: Classifies the anatomical region (head & neck, chest, or abdomen and pelvis)
3. **Multi-headed Anatomical Location-Specific Classifiers**: Three separate classifiers trained for each anatomical region
4. **Weighted Aggregator**: Combines outputs using weights derived from the location estimator

```
├── datamodule.py      # PyTorch Lightning DataModule for handling dataset operations
├── dataset.py         # Custom dataset class for PPGL CT images
├── loss.py            # Implementation of Tversky Loss and Supervised Contrastive Loss
├── main.py            # Main training script
├── model.py           # Model architecture implementation
└── README.md          # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning 2.5.0+
- torchvision
- scikit-image
- numpy

## Dataset

The dataset used in this study comprises 650 axial slices extracted from the CE-CTs of 287 human subjects[1]. The dataset is categorized based on genetic clusters:

- SDHx (362 samples)
- VHL/EPAS1 (75 samples)
- Kinase signaling (99 samples)
- Sporadic (114 samples)

The anatomical location-wise composition includes:

- Abdomen (506 samples)
- Chest (80 samples)
- Head and neck (64 samples)

## Usage

### Training

To train the model using 10-fold cross-validation:

```bash
python main.py
```

This will:

1. Load and prepare the dataset
2. Train the model for each fold
3. Save the best model for each fold
4. Output the results to `results.json`

### Custom Training

You can modify the hyperparameters in `main.py`:

```python
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
EPOCHS = 200
```

### Inference

To perform inference on new CT images:

```python
import torch
from model import Classifier
from torchvision import transforms
from PIL import Image

# Load model
model = Classifier(lr_dino=1e-5, lr_class=1e-2, weight_decay=0.0005, k=0)
model.load_state_dict(torch.load("model_0.pt"))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584])
])

img = Image.open("path_to_image.tiff").convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Create dummy region tensor (not used during inference)
dummy_region = torch.zeros((1, 4))

# Run inference
with torch.no_grad():
    outputs, reg_pred = model(img_tensor, dummy_region, train=False)
    
    # Get predicted class
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Map to genetic cluster
    genetic_clusters = ["Kinase Signaling", "SDHx", "Sporadic", "VHL/EPAS1"]
    print(f"Predicted genetic cluster: {genetic_clusters[predicted_class]}")
```

## Results

The model achieves significant improvements across multiple genetic types with an overall AUC of 0.722[1]. It outperforms existing state-of-the-art methods by 17.4% and 23.9% in overall F1-score and recall respectively. 

The proposal shows a remarkable increase of 22.5% and 18% in F1-score and recall for kinase signaling, surpassing all prior methods.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{anonymous2025multiheaded,
  title={Multi-Headed Deep Feature Learning for Genetic Clusters Identification of Pheochromocytoma and Paraganglioma from CTs},
  author={Anonymous},
  booktitle={Anonymous Conference},
  year={2025}
}
```

## License

MIT License