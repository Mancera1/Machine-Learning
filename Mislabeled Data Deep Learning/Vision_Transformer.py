"""
Transformer Class & Helper Functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import types
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from google.colab import drive


from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

drive.mount('/content/drive')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def find_last_attention_block_with_patches(vit_model, dummy_input_shape=(1,3,224,224)):
    """
    Runs a dummy forward pass to find the last MHA with seq_len>1.
    NOTE: We do NOT patch the modules here because that can lead to double-patching.
    Instead, we do a small forward hook that just checks the shape of MHA.
    """
    hooks = []
    mha_modules = []

    class TempHook:
        def __init__(self):
            self.attn = None
        def hook_fn(self, module, inp, out):
            # out is the MHA output, not the attn.
            pass

    for module in vit_model.modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_modules.append(module)
            h = TempHook()
            hooks.append((module.register_forward_hook(h.hook_fn), h))

    def shape_check_hook(module, inputs, output):
        q = inputs[0]
        if len(q.shape) == 3:
            if q.shape[0] > 1 or q.shape[1] > 1:
                # Then it's definitely seeing patches
                # We'll store that in a dictionary
                module._seq_len_gt1 = True
            else:
                module._seq_len_gt1 = False
        else:
            module._seq_len_gt1 = False

    # Re-do the hooking with shape_check_hook
    for handle, _ in hooks:
        handle.remove()  # remove previous
    hooks.clear()

    for module in mha_modules:
        h = module.register_forward_hook(shape_check_hook)
        hooks.append(h)

    # dummy pass
    vit_model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(dummy_input_shape, device=device)
        _ = vit_model(dummy_input)

    # Among all MHA modules, find the last one that had ._seq_len_gt1 = True
    last_valid_module = None
    for module in mha_modules:
        if getattr(module, "_seq_len_gt1", False):
            last_valid_module = module

    # cleanup
    for h in hooks:
        h.remove()

    return last_valid_module

def force_return_attn_forward_custom(self, query, key, value, **kwargs):
    """
    Custom forward for nn.MultiheadAttention that always captures 'attn'
    in self.saved_attn by calling the original forward with need_weights=True.
    """
    kwargs.pop("need_weights", None)

    # original forward, ensuring need_weights=True
    out, attn = self._original_forward(query, key, value, need_weights=True, **kwargs)

    # Store the attention weights on the module itself
    self.saved_attn = attn.detach().cpu()  # shape: [batch, n_heads, tgt_len, src_len]
    return out, attn

def patch_vit_attention_to_return(vit_model):
    """
    Patches all nn.MultiheadAttention modules in the given ViT model
    so that they store attention weights (self.saved_attn).
    """
    for module in vit_model.modules():
        if isinstance(module, nn.MultiheadAttention):
            # Save the original forward so we can wrap it
            module._original_forward = module.forward
            # Assign our custom forward
            module.forward = types.MethodType(force_return_attn_forward_custom, module)



class StoreAttentionHook:
    """
    Hook class to store the final attn_weights after forward pass in the last MHA block.
    """
    def __init__(self):
        self.attn_weights = None

    def hook_fn(self, module, input, output):
        # In custom forward, stored attn in module.saved_attn
        self.attn_weights = getattr(module, "saved_attn", None)

def find_last_attention_block(vit_model):
    """
    Looks for the last nn.MultiheadAttention inside vit_model.encoder.layers,
    falling back to iterating over modules if needed.
    """
    if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layers'):
        for block in reversed(vit_model.encoder.layers):
            if hasattr(block, 'self_attn'):
                return block.self_attn
    # Fallback
    last = None
    for module in vit_model.modules():
        if isinstance(module, nn.MultiheadAttention):
            last = module
    return last

def attach_attention_hook(vit_model):
    last_attention = find_last_attention_block_with_patches(vit_model)
    if last_attention is None:
        raise RuntimeError("No MultiheadAttention with seq_len>1 found.")
    hook = StoreAttentionHook()
    last_attention.register_forward_hook(hook.hook_fn)
    return hook


class CIFARCustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert CHW to HWC for PIL compatibility
        img = self.data[idx].transpose(1, 2, 0)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def initialize_vit_model(num_classes):
    vit = models.vit_b_16(pretrained=True)
    vit.heads = nn.Linear(in_features=768, out_features=num_classes)
    return vit.to(device)

def introduce_label_noise(labels, num_classes, noise_ratio):
    num_samples = len(labels)
    num_corrupt = int(noise_ratio * num_samples)
    indices = np.random.choice(num_samples, num_corrupt, replace=False)
    corrupted_labels = labels.copy()
    corrupted_labels[indices] = np.random.randint(0, num_classes, size=num_corrupt)
    return corrupted_labels

# Define transforms for CIFAR and MNIST
cifar_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=4),
    RandAugment(num_ops=3, magnitude=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mnist_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert 1 channel to 3.
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class ViTTrainer:
    def __init__(self,
                 dataset,
                 noise_ratio=0.0,
                 batch_size=64,
                 root_model_dir="/content/drive/MyDrive/vit_models",
                 resume=True,
                 mixup=False,
                 mixup_alpha=1.0): # Beta distribution parameter
        """
        Initialize the trainer.

        Args:
          dataset (str): One of "cifar10", "cifar100", or "mnist".
          noise_ratio (float): Noise level (0.0, 0.1, 0.2, etc.).
                               A value >0 triggers label noise.
          batch_size (int): Batch size for DataLoaders.
          root_model_dir (str): Base directory where models are saved.
          resume (bool): If True (default), load existing model weights if available;
                         if False, start with a new model regardless of saved weights.
          mixup (bool): If True, apply Mixup during training. Default=False (off).
          mixup_alpha (float): Alpha parameter for the Beta distribution in Mixup. Default=1.0.
        """
        self.dataset = dataset.lower()
        self.noise_ratio = noise_ratio
        self.batch_size = batch_size
        self.root_model_dir = root_model_dir
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha

        # Map noise ratio to subfolder name
        if self.noise_ratio == 0:
            self.subfolder = "base"
        else:
            percent = int(self.noise_ratio * 100)
            self.subfolder = f"{percent}percent"
        self.save_dir = os.path.join(self.root_model_dir, self.subfolder)
        os.makedirs(self.save_dir, exist_ok=True)

        # dataset-specific parameters
        if self.dataset == "cifar10":
            self.num_classes = 10
            self.model_file = "vit_cifar10.pth"
        elif self.dataset == "cifar100":
            self.num_classes = 100
            self.model_file = "vit_cifar100.pth"
        elif self.dataset == "mnist":
            self.num_classes = 10
            self.model_file = "vit_mnist.pth"
        else:
            raise ValueError("Dataset must be one of 'cifar10', 'cifar100', or 'mnist'")
        self.model_path = os.path.join(self.save_dir, self.model_file)

        # Initialize model (load weights if resume==True and a saved model exists)
        self.model = initialize_vit_model(self.num_classes)
        if resume and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded existing model from {self.model_path}")
        else:
            print("No existing model found or forced training from scratch; initializing new model.")
            print(f"{self.model_path}")

        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Load dataset and create DataLoaders
        self.load_data()

    def _mixup_data(self, x, y, alpha=1.0):
        """
        Returns mixed inputs, pairs of targets, and lambda.
        If alpha=0, this degrades to just the original data.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def load_data(self):
        """
        Load dataset and create DataLoaders with appropriate transforms and noise.
        """
        if self.dataset == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
            test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
            train_data = np.array(train_set.data).transpose(0, 3, 1, 2)
            train_labels = np.array(train_set.targets)
            test_data = np.array(test_set.data).transpose(0, 3, 1, 2)
            test_labels = np.array(test_set.targets)
            if self.noise_ratio > 0:
                train_labels = introduce_label_noise(train_labels, num_classes=10, noise_ratio=self.noise_ratio)
            self.train_dataset = CIFARCustomDataset(train_data, train_labels, transform=cifar_transform)
            self.test_dataset  = CIFARCustomDataset(test_data, test_labels, transform=cifar_transform)

        elif self.dataset == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
            test_set  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
            train_data = np.array(train_set.data).transpose(0, 3, 1, 2)
            train_labels = np.array(train_set.targets)
            test_data = np.array(test_set.data).transpose(0, 3, 1, 2)
            test_labels = np.array(test_set.targets)
            if self.noise_ratio > 0:
                train_labels = introduce_label_noise(train_labels, num_classes=100, noise_ratio=self.noise_ratio)
            self.train_dataset = CIFARCustomDataset(train_data, train_labels, transform=cifar_transform)
            self.test_dataset  = CIFARCustomDataset(test_data, test_labels, transform=cifar_transform)

        elif self.dataset == "mnist":
            train_set = torchvision.datasets.MNIST(root="./data", train=True, transform=mnist_transform, download=True)
            test_set  = torchvision.datasets.MNIST(root="./data", train=False, transform=mnist_transform, download=True)
            # For MNIST, inject noise if needed (labels are stored in a tensor)
            train_labels = train_set.targets.numpy()
            if self.noise_ratio > 0:
                train_labels = introduce_label_noise(train_labels, num_classes=10, noise_ratio=self.noise_ratio)
                train_set.targets = torch.from_numpy(train_labels)
            self.train_dataset = train_set
            self.test_dataset  = test_set

        else:
            raise ValueError("Dataset must be one of 'cifar10', 'cifar100', or 'mnist'")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"DataLoaders initialized for {self.dataset}")

    def load_history(self):
        """Load the training history from disk if it exists."""
        history_path = self.model_path.replace(".pth", "_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
            print("Loaded history from", history_path)
            return history
        else:
            print("No history file found at", history_path)
            return None

    def save_history(self, history):
        """Save training history to disk."""
        history_path = self.model_path.replace(".pth", "_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(history, f)
        print("Training history saved to", history_path)

    def train(self, num_epochs=5):
        """
        Train (or continue training) the model for a given number of epochs.
        Loads existing history (if any), trains the model, then saves both model and updated history.
        """
        history = self.load_history()
        if history is None:
            history = {
                "loss": [], "val_loss": [],
                "accuracy": [], "val_accuracy": [],
                "val_f1": [], "val_recall": [], "val_precision": []
            }

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                # Mixup logic
                if self.mixup:
                    images, labels_a, labels_b, lam = self._mixup_data(images, labels, self.mixup_alpha)
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self.mixup:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels_a).sum().item() * lam
                    correct += (predicted == labels_b).sum().item() * (1 - lam)
                    total += labels.size(0)
                else:
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100.0 * correct / total

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_epoch_loss = val_loss / len(self.test_loader)
            val_epoch_acc = 100.0 * val_correct / val_total

            # Compute additional metrics
            f1 = f1_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted')

            # Append metrics to history
            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_epoch_loss)
            history["accuracy"].append(epoch_acc)
            history["val_accuracy"].append(val_epoch_acc)
            history["val_f1"].append(f1)
            history["val_recall"].append(recall)
            history["val_precision"].append(precision)

            print(f"\nEpoch [{epoch+1}/{num_epochs}] - "
                  f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% | "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%\n"
                  f"Val F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved to", self.model_path)
        self.save_history(history)
        return history

    def continue_training(self, additional_epochs=5):
        """
        Continue training the model for additional epochs. The training history is loaded,
        updated with loss, accuracy, and additional metrics (F1, recall, precision),
        and then saved.
        """
        history = self.load_history()
        if history is None:
            print("No existing history found; starting fresh training.")
            history = {
                "loss": [], "val_loss": [],
                "accuracy": [], "val_accuracy": [],
                "val_f1": [], "val_recall": [], "val_precision": []
            }

        for epoch in range(additional_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            progress_bar = tqdm(self.train_loader, desc=f"Continued Training - Epoch {epoch+1}/{additional_epochs}")

            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                # Mixup logic
                if self.mixup:
                    images, labels_a, labels_b, lam = self._mixup_data(images, labels, self.mixup_alpha)
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Training accuracy calculation
                if self.mixup:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels_a).sum().item() * lam
                    correct += (predicted == labels_b).sum().item() * (1 - lam)
                    total += labels.size(0)
                else:
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100.0 * correct / total

            # Validation phase (unchanged)
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_epoch_loss = val_loss / len(self.test_loader)
            val_epoch_acc = 100.0 * val_correct / val_total

            # Compute additional metrics
            f1 = f1_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted')

            # Append metrics to history
            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_epoch_loss)
            history["accuracy"].append(epoch_acc)
            history["val_accuracy"].append(val_epoch_acc)
            history["val_f1"].append(f1)
            history["val_recall"].append(recall)
            history["val_precision"].append(precision)

            print(f"\nEpoch [{epoch+1}/{additional_epochs}] - "
                  f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% | "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%\n"
                  f"Val F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

        torch.save(self.model.state_dict(), self.model_path)
        print("Updated model saved to", self.model_path)
        self.save_history(history)
        return history

    def plot_metrics(self):
        """
        Plot the training history (accuracy & loss), compute classification metrics on the test set,
        and display an overfitting indicator (difference between validation and training loss).
        """
        history = self.load_history()
        if history is None:
            print("No training history available to plot.")
            return

        plt.figure(figsize=(14, 12))

        # Accuracy plot
        plt.subplot(3, 2, 1)
        plt.plot(history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(history['val_accuracy'], 'b', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        # Loss plot
        plt.subplot(3, 2, 2)
        plt.plot(history['loss'], 'r', label='Training Loss')
        plt.plot(history['val_loss'], 'b', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot additional stored metrics
        plt.subplot(3, 2, 3)
        plt.plot(history['val_f1'], 'g', label='Val F1 Score')
        plt.plot(history['val_recall'], 'orange', label='Val Recall')
        plt.plot(history['val_precision'], 'purple', label='Val Precision')
        plt.title('Additional Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()

        # Overfitting indicator: difference between validation and training loss
        loss_diff = np.array(history['val_loss']) - np.array(history['loss'])
        plt.subplot(3, 2, 4)
        plt.plot(loss_diff, 'magenta')
        plt.title('Val - Training Loss Difference (Overfitting Indicator)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Difference')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        if not getattr(trainer, "_ece_panel_added", False):

            orig_plot = trainer.plot_metrics

            def _ece_plot(self, *a, **kw):
                orig_plot(*a, **kw)
                hist = self.load_history()
                if hist and "val_ece" in hist:
                    plt.figure(figsize=(4, 3))
                    plt.plot(hist["val_ece"], 'c', label='Val ECE')
                    plt.title('Expected Calibration Error')
                    plt.xlabel('Epochs');  plt.ylabel('ECE')
                    plt.grid(True);  plt.legend();  plt.show()

            # re‑bind the patched function
            trainer.plot_metrics = types.MethodType(_new_plot, trainer)
            trainer._ece_panel_added = True


    def confusion_matrix(self):
        """
        Compute and display the confusion matrix on the test set,
        auto‐resizing only if we have >20 classes.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        n_classes = cm.shape[0]

        # choose figure size based on number of classes
        if n_classes > 50:
            figsize = (24, 24)
            label_fontsize = 4
        elif n_classes > 20:
            figsize = (12, 12)
            label_fontsize = 6
        else:
            figsize = (8, 6)
            label_fontsize = 8

        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(
            ax=ax,
            cmap=plt.cm.Blues,
            xticks_rotation='vertical',   # rotate x labels
            values_format='d'             # integer format
        )

        # shrink font if many classes
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_fontsize)

        title = (
            f"Confusion Matrix: {self.subfolder} Noise"
            if self.subfolder != 'base'
            else "Confusion Matrix"
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


    def visualize_attention(self, dataset_index=0, head_idx=0):
        """
        Visualize how the [CLS] token in the last attention block attends to the patch tokens
        as a 2D heatmap (e.g. 14x14). We also handle the case where the attention shape
        comes back [B, T, T] (missing a head dimension).
        """
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt

        # patch once
        if not hasattr(self, "_attention_patched") or not self._attention_patched:
            patch_vit_attention_to_return(self.model)
            self._attention_patched = True

        # Find the last MultiheadAttention block
        last_attention = find_last_attention_block(self.model)
        if last_attention is None:
            print("No MultiheadAttention found in this ViT model.")
            return

        # Create a small hook to read out the saved_attn
        class TempHook:
            def __init__(self):
                self.attn = None
            def hook_fn(self, module, inp, out):
                self.attn = getattr(module, "saved_attn", None)

        hook_obj = TempHook()
        hook_handle = last_attention.register_forward_hook(hook_obj.hook_fn)

        # Get one image from the test dataset, forward pass
        sample_img, sample_label = self.test_dataset[dataset_index]
        sample_img = sample_img.unsqueeze(0).to(device)  # shape [1, C, H, W]

        self.model.eval()
        with torch.no_grad():
            _ = self.model(sample_img)

        # Retrieve attention: could be [batch, n_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
        attn_weights = hook_obj.attn
        hook_handle.remove()  # remove hook

        if attn_weights is None:
            print("No attention was captured. Possibly the block doesn't see patch tokens.")
            return

        # If[B, T, T], interpret as [B, 1, T, T] (n_heads=1)
        if attn_weights.dim() == 3:
            # shape: [batch, seq_len, seq_len]
            attn_weights = attn_weights.unsqueeze(1)

        if attn_weights.ndim != 4:
            print(f"Unexpected attention shape (still): {attn_weights.shape}")
            return

        if attn_weights.size(0) < 1:
            print(f"No batch dimension: {attn_weights.shape}")
            return

        attn_weights = attn_weights[0]

        if attn_weights.size(-1) <= 1:
            print("This final attention block has seq_len=1 (only CLS). No patch tokens to visualize.")
            return

        if head_idx >= attn_weights.shape[0]:
            print(f"head_idx={head_idx} out of range. #Heads = {attn_weights.shape[0]}")
            return

        cls_row = attn_weights[head_idx, 0]
        patch_attn = cls_row[1:]
        num_patches = patch_attn.shape[0]

        side = int(num_patches**0.5)
        if side * side != num_patches:
            print(f"Cannot reshape {num_patches} tokens into a square (e.g. 14x14).")
            return

        patch_map = patch_attn.reshape(side, side).cpu().numpy()

        # attention map
        plt.figure(figsize=(6,5))
        plt.imshow(patch_map, cmap='viridis')
        plt.title(f"[CLS] → Patch Attention (Head {head_idx})\nLabel={sample_label}")
        plt.colorbar()
        plt.show()

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        unnorm = sample_img * std + mean
        np_img = unnorm.squeeze(0).permute(1,2,0).cpu().numpy().clip(0,1)

        plt.figure(figsize=(4,4))
        plt.imshow(np_img)
        plt.title(f"Original Image (Label={sample_label})")
        plt.axis('off')
        plt.show()

from __future__ import annotations

import types, math, inspect
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# 1. VISUALISATION OF LABEL–NOISE IMPACT: ATTENTION ROLLOUT – works with nn.MultiheadAttention OR #
# torchvision.models.vision_transformer.Attention

def _patch_module_for_attn(module: torch.nn.Module):
    """
    Wrap module whose forward signature contains need_weights
    so that it always returns—and stores—per-head attention.
    """
    if getattr(module, "_attn_patched", False):
        return

    sig = inspect.signature(module.forward)
    if "need_weights" not in sig.parameters:
        return

    module._orig_forward = module.forward # original

    def _new_forward(self, *args, **kwargs):
        kwargs["need_weights"] = True
        # keep per-head detail
        if "average_attn_weights" in sig.parameters:
            kwargs["average_attn_weights"] = False
        out = self._orig_forward(*args, **kwargs)
        attn_w = out[1] if isinstance(out, tuple) else out.attn_output_weights
        self.saved_attn = attn_w.detach()# [B, h, T, T] | [B,T,T]
        return out

    module.forward = types.MethodType(_new_forward, module)
    module._attn_patched = True


def attention_rollout(model: torch.nn.Module,
                      image: torch.Tensor,
                      discard_ratio: float = 0.0,
                      head_fusion: str = "mean") -> np.ndarray:
    """
    Compute Abnar & Zuidema attention‐rollout heat-map.
    Compatible with both torchvision and timm ViTs.
    """
    # patch eligible sub-module once
    for m in model.modules():
        _patch_module_for_attn(m)

    # forward pass
    _ = model.eval()(image)

    # attention matrices
    attn_mats = []
    for m in model.modules():
        if hasattr(m, "saved_attn"):
            A = m.saved_attn
            if A.dim() == 4:                      # [B, h, T, T]
                A = A.mean(1)                     # fuse heads
            if A.size(0) == 1:                    # squeeze batch
                A = A[0]                          # [T, T]
            A = A / A.sum(-1, keepdim=True)       # row-norm
            attn_mats.append(A + torch.eye(A.size(-1), device=A.device))


    if not attn_mats:
        raise RuntimeError("No attention weights captured – "
                           "ViT implementation differs; check patching.")

    # 4) multiply matrices
    rollout = attn_mats[0]
    for A in attn_mats[1:]:
        rollout = A @ rollout
    cls_vec = rollout[0, 1:]                # CLS → patches
    side = int(math.sqrt(cls_vec.numel()))
    return cls_vec.reshape(side, side).cpu().numpy()


def _accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct / total


def estimate_accuracy_vs_noise(trainer,  # existing ViTTrainer instance
                               noise_levels: List[float] = None,
                               subset: int | None = 5000,
                               batch_size: int = 128,
                               cache_dir: str | None = None) -> Dict[float, float]:
    """Measure *test* accuracy after retraining with synthetic noise levels.

    The function *re‑uses* existing trained checkpoints if they exist under
    trainer.root_model_dir/<noise>%/vit_*.pth, otherwise it will *train for 1
    epoch only* (quick diagnostic) to obtain a robustness curve.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]

    acc_dict = {}
    base_kwargs = dict(dataset=trainer.dataset,
                       batch_size=batch_size,
                       root_model_dir=trainer.root_model_dir,
                       resume=True)

    for nl in noise_levels:
        tmp_trainer = trainer.__class__(noise_ratio=nl, **base_kwargs)
        if not os.path.exists(tmp_trainer.model_path):
            # *Minimal* training – 1 epoch only – fast diagnostic
            tmp_trainer.train(num_epochs=1)
        acc = _accuracy(tmp_trainer.test_loader.dataset if subset is None else
                         DataLoader(Subset(tmp_trainer.test_dataset, range(subset)),
                                    batch_size=batch_size),
                         tmp_trainer.model, trainer.model.device)
        acc_dict[nl] = acc
    return acc_dict


def expected_calibration_error(logits: torch.Tensor,
                               labels: torch.Tensor,
                               n_bins: int = 15) -> float:
    """Compute ECE (Guo et al., 2017) on (logits, labels) batch‑wise."""
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].float().mean()
            ece += (bin_conf - bin_acc).abs() * mask.float().mean()
    return ece.item()


class MislabelDetector:
    """Flag suspicious samples *without* access to ground‑truth by combining
    (i) confidence filtering, (ii) augmentation‑consistency, and (iii) ensemble
    agreement.  Instantiate with *one* reference model, then call the methods
    in sequence.  Pass‑through functions are provided so that the detector can
    be slotted into an existing notebook with minimal refactor.
    """

    def __init__(self, model: torch.nn.Module, dataset, device=None):
        self.model = model.eval()
        self.dataset = dataset
        self.device = device or next(model.parameters()).device##

    def confidence_scores(self, loader: DataLoader, batch_size: int = 256) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                conf, _ = F.softmax(logits, 1).max(dim=1)
                scores.extend(conf.cpu().numpy())
        return np.asarray(scores)


    def _single_aug(self, x):
        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(x.size(-1), padding=4)
        ])
        return aug(x)

    def consistency_scores(self, indices: List[int], n_aug: int = 4, batch_size: int = 128) -> np.ndarray:
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        inconsistencies = []
        with torch.no_grad():
            for x, _ in loader:
                preds = []
                for _ in range(n_aug):
                    x_aug = self._single_aug(x)
                    logits = self.model(x_aug.to(self.device))
                    preds.append(logits.argmax(1).cpu())
                preds = torch.stack(preds)  # [n_aug,B]
                mode = torch.mode(preds, dim=0).values  # majority vote
                inc = (preds != mode).float().mean(dim=0)  # [B] fraction of mismatches
                inconsistencies.extend(inc.numpy())
        return np.asarray(inconsistencies)


    @staticmethod
    def ensemble_agreement(models: List[torch.nn.Module], loader: DataLoader, device=None) -> np.ndarray:
        """Return 1‑agreement scores (lower ⇒ less agreement) per sample."""
        device = device or next(models[0].parameters()).device
        for m in models:
            m.eval().to(device)
        agreements = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                votes = [m(x).argmax(1).cpu() for m in models]
                votes = torch.stack(votes)               # [M,B]
                mode = torch.mode(votes, dim=0).values   # [B]
                disagree = (votes != mode).float().mean(dim=0)  # [B]
                agreements.extend(disagree.numpy())
        return np.asarray(agreements)


    def rank_suspects(self,
                      loader: DataLoader,
                      ensemble_models: List[torch.nn.Module] | None = None,
                      low_conf_thresh: float = 0.2,
                      top_k: int | None = 1000) -> List[int]:
        """Return indices (w.r.t *self.dataset*) ranked by likelihood of mis‑label."""
        N = len(self.dataset)
        indices = list(range(N))
        conf = self.confidence_scores(loader)
        conf_risk = 1 - conf                       # high risk if low confidence

        aug_inc = self.consistency_scores(indices)
        risk = conf_risk + aug_inc                 # combine (simple sum)

        if ensemble_models is not None:
            ens_loader = DataLoader(self.dataset, batch_size=loader.batch_size, shuffle=False)
            ens_disa = self.ensemble_agreement(ensemble_models, ens_loader, self.device)
            risk += ens_disa

        ranked = np.argsort(-risk)  # descending risk
        suspects = ranked[:top_k] if top_k is not None else ranked
        return suspects.tolist()


# Noise Robustness
def _quick_accuracy(model, loader):
    model.eval(); tot=cor=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            cor += (model(x).argmax(1)==y).sum().item()
            tot += y.size(0)
    return cor/tot

def evaluate_noise_curve(base_trainer, noise_levels=[0,.1,.2,.4,.6]):
    curve = {}
    common = dict(dataset=base_trainer.dataset,
                  batch_size=base_trainer.batch_size,
                  root_model_dir=base_trainer.root_model_dir,
                  resume=True)
    for nl in noise_levels:
        tr = base_trainer.__class__(noise_ratio=nl, **common)
        if not os.path.exists(tr.model_path):
            tr.train(num_epochs=1)
        acc = _quick_accuracy(tr.model, tr.test_loader)
        curve[nl] = acc
    return curve


def fine_tune_one_epoch(trainer,
                        corrected_indices: List[int],
                        batch_size: int = 64,
                        lr: float = 5e-6,
                        patience: int = 3):
    """
    Perform single‑epoch fine‑tuning after relabelling or filtering.
    """
    subset = Subset(trainer.train_dataset, corrected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    old_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    optim = torch.optim.AdamW(trainer.model.parameters(), lr=lr)

    best_val_acc, trigger_times = -1, 0
    for epoch in range(100):
        trainer.model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = trainer.model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()

        val_acc = calculate_accuracy(trainer.test_loader, trainer.model, trainer.model.device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
    print(f"\nOne‑epoch fine‑tune finished – best val acc: {best_val_acc:.2%}")

    ft_path = trainer.model_path.replace('.pth', '_finetuned.pth')
    torch.save(trainer.model.state_dict(), ft_path)
    print("Fine‑tuned weights saved to", ft_path)


def extend_trainer(trainer):
    trainer.attention_rollout = lambda img, **kw: attention_rollout(trainer.model, img.to(trainer.model.device), **kw)
    trainer.evaluate_noise_curve = lambda *a, **kw: estimate_accuracy_vs_noise(trainer, *a, **kw)
    trainer.compute_ece = lambda: _compute_ece_dataset(trainer)
    trainer.detector = MislabelDetector(trainer.model, trainer.train_dataset, trainer.model.device)
    trainer.fine_tune = lambda idx, **kw: fine_tune_one_epoch(trainer, idx, **kw)
    return trainer


def _compute_ece_dataset(trainer, n_bins: int = 15):
    logits_list, labels_list = [], []
    trainer.model.eval()
    with torch.no_grad():
        for x, y in trainer.test_loader:
            x, y = x.to(trainer.model.device), y.to(trainer.model.device)
            logits_list.append(trainer.model(x))
            labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    return expected_calibration_error(logits, labels, n_bins)