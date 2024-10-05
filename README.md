# Unsupervised Early Exit for DNN

This repo belongs to TMLR paper An Unsupervised Early Exit Mechanism for Deep Neural Networks

## Prerequisites

Before running this project, ensure you have the following installed on your system:

- **Python** (>= 3.7.1)
- **PyTorch** (>= 1.1.0)
- **Torchvision** (>= 0.3.0)

You can install these dependencies using the following commands:

```bash
pip install torch>=1.1.0 torchvision>=0.3.0
```

Ensure you have the appropriate version of Python installed. If not, you can download it from [Python Official Website](https://www.python.org/downloads/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hoseinkhs/AutoCacheLayer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd project-name
   ```
3. Install the required packages:
   ```bash
   pip install torch>=1.1.0 torchvision>=0.3.0
   ```

## Usage

Once you have set up the environment and installed the dependencies, you can run the scripts. Example usage:

```bash
python train.py \
    --experiment "Cifar10" \
    --num_classes 10 \
    --train_epochs 5 \
    --train_device "cuda:0" \
    --test_device "cuda:0" \
    --data_root "./data/cifar10" \
    --backbone_type "Resnet18" \
    --backbone_conf_file "backbone_conf.yaml" \
    --exit_type "Dense2LayerTemp" \
    --lr 0.1 \
    --batch_size 16 \
```
Or you can easily run the scripts
## License

This project is licensed under the MIT License.

---

Feel free to replace "Project Name" and any placeholder text with your actual project details.
