# Robust Learning Under Label Noise

> **A reference implementation & comparative study of modern noise-robust training techniques for image classification**

## ✨ What’s Inside

| Category | Details |
| -------- | ------- |
| **Techniques** | Active–Passive Loss (**NFL + RCE**), **Co-Teaching**, **Co-Teaching+**, **T-Revision** (transition-matrix estimation), ViT-based label-correction, clean CE baselines |
| **Architectures** | CNN-5, **CNN-8** (Keras & PyTorch), DenseNet-169, ResNet-18/50, **ViT-B/16** |
| **Frameworks** | PyTorch 2.3 • TensorFlow/Keras 2.16 |
| **Datasets** | CIFAR-10/100, MNIST |
| **Noise Model** | *Symmetric* flips @ 0 / 10 / 20 / 30 / 40 / 50 / 60 % |
| **Metrics** | Accuracy, F1 / Precision / Recall, Expected Calibration Error, Diagnostic Index, D-Index |
| **Visuals** | CLS→patch attention maps, confusion matrices, transition-matrix heat-maps, loss/accuracy curves |

> Most prior repos test **one** defence in isolation. **This repo lets you mix & match** filtering (Co-Teaching), gradient shaping (NFL + RCE), transition-matrix correction (T-Revision), and architecture choice — with unified logging.
