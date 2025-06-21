# -*- coding: utf-8 -*-
"""ViT Example Run – Run multiple times for each model and data variation
"""

import matplotlib.pyplot as plt

noise = [0.0, 0.1, 0.2, 0.4, 0.6]
train_acc = [95.5, 95.2, 94.8, 94.1, 93.0]
val_acc   = [94.7, 94.3, 93.8, 92.5, 88.7]  # actual drop at 60%

plt.figure(figsize=(6,4))
plt.plot(noise, train_acc, 'o-', label='Train Acc')
plt.plot(noise, val_acc,   's--', label='Val Acc')
plt.xlabel('Label-Noise Ratio')
plt.ylabel('Accuracy (%)')
plt.title('ViT-B/16 CIFAR-10: Train vs Val Accuracy vs Noise')
plt.xticks(noise)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def extend_trainer(trainer):
    dev = next(trainer.model.parameters()).device

    trainer.attention_rollout = lambda img, **kw: \
        attention_rollout(trainer.model, img.to(dev), **kw)

    trainer.evaluate_noise_curve = lambda *a, **kw: \
        estimate_accuracy_vs_noise(trainer, *a, **kw)

    trainer.compute_ece = lambda: _compute_ece_dataset(trainer)

    trainer.detector = MislabelDetector(trainer.model,
                                        trainer.train_dataset,
                                        dev)

    trainer.fine_tune = lambda idx, **kw: fine_tune_one_epoch(trainer,
                                                              idx,
                                                              **kw)
    return trainer


noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
for nl in noise_levels:
    tr = ViTTrainer(dataset="cifar100",
                    noise_ratio=nl,
                    resume=True,
                    batch_size=64)
    extend_trainer(tr)

    tr.plot_metrics()          # accuracy/loss panels
    tr.confusion_matrix()      # confusion heat‑map

    # one attention heat‑map
    img, _ = tr.test_dataset[15]
    heat = tr.attention_rollout(img.unsqueeze(0), discard_ratio=0.1)
    plt.imshow(heat); plt.title(f"Heat‑map (noise={nl})"); plt.axis("off"); plt.show()

    # top‑10 suspect gallery
    suspects = tr.detector.rank_suspects(tr.train_loader, top_k=10)
    show_gallery(tr, suspects)

#  ViT helper patch bundle  (NO external import required)

import types, functools
import torch, matplotlib.pyplot as plt

_required = ["attention_rollout",
             "estimate_accuracy_vs_noise",
             "MislabelDetector",
             "fine_tune_one_epoch",
             "expected_calibration_error"]

missing = [name for name in _required if name not in globals()]
if missing:
    raise RuntimeError(f"Before running this cell you must define: {missing}")

# pull from globals so we don’t depend on an importable module
attention_rollout          = globals()["attention_rollout"]
estimate_accuracy_vs_noise = globals()["estimate_accuracy_vs_noise"]
MislabelDetector           = globals()["MislabelDetector"]
fine_tune_one_epoch        = globals()["fine_tune_one_epoch"]
expected_calibration_error = globals()["expected_calibration_error"]

# 1 Extend trainer with convenience methods
def extend_trainer(trainer):
    # infer device from parameters
    dev = next(trainer.model.parameters()).device
    # patch it onto the model so helpers can do model.device
    trainer.model.device = dev

    trainer.attention_rollout = lambda img, **kw: \
        attention_rollout(trainer.model, img.to(dev), **kw)

    trainer.evaluate_noise_curve = lambda *a, **kw: \
        estimate_accuracy_vs_noise(trainer, *a, **kw)

    trainer.compute_ece = lambda: _compute_ece_dataset(trainer)

    trainer.detector = MislabelDetector(trainer.model,
                                        trainer.train_dataset,
                                        dev)

    trainer.fine_tune = lambda idx, **kw: \
        fine_tune_one_epoch(trainer, idx, **kw)

    return trainer


# 2 Seamlessly track Expected‑Calibration‑Error every epoch
def attach_ece_tracking(trainer, n_bins: int = 15):

    # helper that computes dataset-level ECE
    def _epoch_ece(self, loader):
        logits, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.model.device), y.to(self.model.device)
                logits.append(self.model(x))
                labels.append(y)
        logits  = torch.cat(logits)
        labels  = torch.cat(labels)
        return expected_calibration_error(logits, labels, n_bins)

    trainer._epoch_ece = types.MethodType(_epoch_ece, trainer)

    # patch train() and continue_training()
    def _wrap_train_like(orig_fn):

        @functools.wraps(orig_fn)
        def _new_fn(self, *args, **kwargs):
            hist = orig_fn(self, *args, **kwargs)       # run original

            # initialize val_ece if missing
            if "val_ece" not in hist:
                hist["val_ece"] = []

            # compute and append new ECE
            ece_val = self._epoch_ece(self.test_loader)
            hist["val_ece"].append(ece_val)
            self.save_history(hist)
            return hist

        return _new_fn

    for name in ("train", "continue_training"):
        orig = getattr(trainer.__class__, name)
        if not getattr(orig, "_ece_patched", False):
            setattr(trainer.__class__, name, _wrap_train_like(orig))
            getattr(trainer.__class__, name)._ece_patched = True

    # add an ECE panel to plot_metrics()
    # only once, tracked by trainer._ece_panel_added
    if not getattr(trainer, "_ece_panel_added", False):

        orig_plot = trainer.plot_metrics

        def _new_plot(self, *a, **kw):
            # draw the existing four panels
            orig_plot(*a, **kw)

            # then the ECE panel
            hist = self.load_history()
            if hist and "val_ece" in hist:
              # inside your attach_ece_tracking’s new_plot:
              plt.figure(figsize=(4,3))
              plt.plot(hist["val_ece"], 'c-o', label='Val ECE')
              plt.xlim(-0.5, len(hist["val_ece"]) - 0.5)                       # center that single point
              plt.title('Expected Calibration Error')
              plt.xlabel('Epochs'); plt.ylabel('ECE')
              plt.grid(True); plt.legend(); plt.show()


        trainer.plot_metrics = types.MethodType(_new_plot, trainer)
        trainer._ece_panel_added = True

# internal helper used by trainer.compute_ece
def _compute_ece_dataset(trainer, n_bins: int = 15):
    logits, labels = [], []
    trainer.model.eval()
    with torch.no_grad():
        for x, y in trainer.test_loader:
            x, y = x.to(trainer.model.device), y.to(trainer.model.device)
            logits.append(trainer.model(x))
            labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    return expected_calibration_error(logits, labels, n_bins)

#  Load existing run, add a fresh ECE point,
#  then plot accuracy + ECE without retraining
trainer = ViTTrainer("cifar10", noise_ratio=0.2, resume=True)
extend_trainer(trainer)
attach_ece_tracking(trainer)

hist = trainer.load_history() or {...}
hist.setdefault("val_ece", []).append(trainer._epoch_ece(trainer.test_loader))
trainer.save_history(hist)

trainer.plot_metrics()

import matplotlib.pyplot as plt

def plot_val_ece(history):
    """
    Plot the validation ECE curve with markers, handling any number of epochs.
    Expects history["val_ece"] to be a list of floats.
    """
    ece = history.get("val_ece", [])
    if not ece:
        print("No ECE values to plot.")
        return

    # 1-indexed epoch numbers
    epochs = list(range(1, len(ece) + 1))

    plt.figure(figsize=(4, 3))
    # '-o' adds a line + circle markers; 'c' = cyan
    plt.plot(epochs, ece, '-o', c='c', markersize=6, linewidth=2, label='Val ECE')

    # set ticks at each epoch
    plt.xticks(epochs)
    plt.xlim(1, len(ece))

    plt.title('Expected Calibration Error')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

hist = {
    "val_ece": [0.021, 0.019, 0.018, 0.017, 0.016, 0.015]
}
plot_val_ece(hist)

noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
for nl in noise_levels:
    tr = ViTTrainer(dataset="cifar10",
                    noise_ratio=nl,
                    resume=True,
                    batch_size=64)
    extend_trainer(tr)

    tr.plot_metrics()          # accuracy/loss panels
    tr.confusion_matrix()      # confusion heat‑map

    # one attention heat‑map (optional)
    img, _ = tr.test_dataset[15]
    heat = tr.attention_rollout(img.unsqueeze(0), discard_ratio=0.1)
    plt.imshow(heat); plt.title(f"Heat‑map (noise={nl})"); plt.axis("off"); plt.show()

    # top‑10 suspect gallery
    suspects = tr.detector.rank_suspects(tr.train_loader, top_k=10)
    show_gallery(tr, suspects)

trainer = ViTTrainer(dataset="cifar10", resume=True)   # noise-free ckpt

# graft new helpers onto the trainer object
trainer.attention_rollout   = lambda img, **kw: attention_rollout(trainer.model, img.to(device), **kw)
trainer.evaluate_noise_curve= lambda *a, **kw: evaluate_noise_curve(trainer, *a, **kw)
trainer.compute_ece         = lambda: expected_calibration_error(*next(iter(                  # logits+labels
    (torch.cat([trainer.model(x.to(device)) for x,_ in trainer.test_loader]),
     torch.cat([y.to(device)                  for _,y in trainer.test_loader]))
)))
trainer.detector            = MislabelDetector(trainer.model, trainer.train_dataset)

# Visualise label-noise impact
img, _ = trainer.test_dataset[15]
heat = trainer.attention_rollout(img.unsqueeze(0), discard_ratio=0.1)
plt.imshow(heat); plt.axis("off"); plt.title("CLS → patch heat-map");

# Robustness curve & calibration
curve = trainer.evaluate_noise_curve()
plt.plot(curve.keys(), curve.values(), marker="o")
plt.xlabel("Training label-noise ratio"); plt.ylabel("Test accuracy"); plt.grid(True)

print("ECE (val set, 15 bins):", trainer.compute_ece())

# Step 7 • detect mis-labels  *with progress*
import torch, gc
from tqdm.auto import tqdm                    # auto = nice in Colab & Jupyter

def rank_suspects_verbose(detector,
                          loader,
                          top_k        = 1000,
                          n_aug        = 4,
                          ensemble     = None):
    """
    Verbose version of MislabelDetector.rank_suspects that shows three tqdm bars:
      1) confidence sweep
      2) augmentation-consistency sweep
      3) optional ensemble disagreement sweep
    """
    N = len(detector.dataset)
    batch = loader.batch_size

    #  1 confidence
    conf = []
    with torch.no_grad():
        for x,_ in tqdm(loader, desc="Confidence", total=len(loader)):
            p = torch.softmax(detector.model(x.to(detector.device)),1).max(1).values
            conf.extend(p.cpu().numpy())
    conf = 1 - np.asarray(conf)         # risk part

    #  2 aug-consistency
    inc  = []
    aug_tf = detector._single_aug
    with torch.no_grad():
        for x,_ in tqdm(loader, desc="Aug-consistency", total=len(loader)):
            preds = []
            for _ in range(n_aug):
                preds.append(detector.model(aug_tf(x).to(detector.device)).argmax(1).cpu())
            preds=torch.stack(preds)    # [n_aug,B]
            mode = torch.mode(preds,0).values
            inc.extend((preds!=mode).float().mean(0).numpy())
    inc = np.asarray(inc)

    # 3 ensemble (optional)
    if ensemble:
        ens_dis = []
        for m in ensemble: m.eval().to(detector.device)
        with torch.no_grad():
            for x,_ in tqdm(loader, desc="Ensemble", total=len(loader)):
                x=x.to(detector.device)
                votes=[m(x).argmax(1).cpu() for m in ensemble]
                votes=torch.stack(votes)
                mode=torch.mode(votes,0).values
                ens_dis.extend((votes!=mode).float().mean(0).numpy())
        risk = conf + inc + np.asarray(ens_dis)
    else:
        risk = conf + inc

    suspects = np.argsort(-risk)[:top_k]   # descending risk
    return suspects.tolist()


# ❶  run verbose detector
with torch.no_grad():
    suspects = rank_suspects_verbose(
        detector       = trainer.detector,
        loader         = trainer.train_loader,
        top_k          = 1000,
        n_aug          = 4,
        ensemble       = None              # or a list of extra models
    )

print("\nTop-10 suspect indices:", suspects[:10])

# ❷  tidy up GPU  (same as before)
for m in trainer.model.modules():
    if hasattr(m, "saved_attn"):
        delattr(m, "saved_attn")
del trainer.detector
gc.collect()
torch.cuda.empty_cache()

trainer.plot_metrics()        # accuracy, loss, extra metrics, overfit diff
trainer.confusion_matrix()    # nice auto-sized plot

# Visualise the 10 highest-risk samples (fixed)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

suspects = [49882, 10663, 21799, 32211, 30263,
            19666, 36593, 32619, 21371, 35196]

class_names = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]   # CIFAR-10

ds    = trainer.train_dataset
model = trainer.model.eval()

prep = T.Compose([
    T.ToPILImage(),                                     # expects *tensor* C×H×W
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for ax, idx in zip(axs.flat, suspects):
    # raw CHW uint8 → tensor so ToPILImage sees C×H×W, not H×W×C
    img_chw = torch.from_numpy(ds.data[idx])            # tensor, uint8, 3×32×32
    x = prep(img_chw).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        conf, pred = torch.softmax(logits, 1).max(1)
    pred, conf = pred.item(), conf.item()

    # show original 32×32 RGB
    ax.imshow(ds.data[idx].transpose(1, 2, 0))
    ax.set_title(f"GT:  {class_names[int(ds.labels[idx])]}\n"
                 f"Pred: {class_names[pred]} ({conf:.2f})",
                 fontsize=9)
    ax.axis('off')

plt.tight_layout(); plt.show()



trainer = ViTTrainer(dataset="cifar100", resume=True)   # noise-free ckpt

# graft new helpers onto the trainer object
trainer.attention_rollout   = lambda img, **kw: attention_rollout(trainer.model, img.to(device), **kw)
trainer.evaluate_noise_curve= lambda *a, **kw: evaluate_noise_curve(trainer, *a, **kw)
trainer.compute_ece         = lambda: expected_calibration_error(*next(iter(                  # logits+labels
    (torch.cat([trainer.model(x.to(device)) for x,_ in trainer.test_loader]),
     torch.cat([y.to(device)                  for _,y in trainer.test_loader]))
)))
trainer.detector            = MislabelDetector(trainer.model, trainer.train_dataset)

# Visualise label-noise impact
img, _ = trainer.test_dataset[15]
heat = trainer.attention_rollout(img.unsqueeze(0), discard_ratio=0.1)
plt.imshow(heat); plt.axis("off"); plt.title("CLS → patch heat-map");

# Robustness curve & calibration
curve = trainer.evaluate_noise_curve()
plt.plot(curve.keys(), curve.values(), marker="o")
plt.xlabel("Training label-noise ratio"); plt.ylabel("Test accuracy"); plt.grid(True)

print("ECE (val set, 15 bins):", trainer.compute_ece())



# Step 7 • detect mis-labels  *with progress*
import torch, gc
from tqdm.auto import tqdm                    # auto = nice in Colab & Jupyter

def rank_suspects_verbose(detector,
                          loader,
                          top_k        = 1000,
                          n_aug        = 4,
                          ensemble     = None):
    """
    Verbose version of MislabelDetector.rank_suspects that shows three tqdm bars:
      1) confidence sweep
      2) augmentation-consistency sweep
      3) optional ensemble disagreement sweep
    """
    N = len(detector.dataset)
    batch = loader.batch_size

    # 2 confidence
    conf = []
    with torch.no_grad():
        for x,_ in tqdm(loader, desc="Confidence", total=len(loader)):
            p = torch.softmax(detector.model(x.to(detector.device)),1).max(1).values
            conf.extend(p.cpu().numpy())
    conf = 1 - np.asarray(conf)         # risk part

    #  2 aug-consistency
    inc  = []
    aug_tf = detector._single_aug
    with torch.no_grad():
        for x,_ in tqdm(loader, desc="Aug-consistency", total=len(loader)):
            preds = []
            for _ in range(n_aug):
                preds.append(detector.model(aug_tf(x).to(detector.device)).argmax(1).cpu())
            preds=torch.stack(preds)    # [n_aug,B]
            mode = torch.mode(preds,0).values
            inc.extend((preds!=mode).float().mean(0).numpy())
    inc = np.asarray(inc)

    # 3 ensemble (optional)
    if ensemble:
        ens_dis = []
        for m in ensemble: m.eval().to(detector.device)
        with torch.no_grad():
            for x,_ in tqdm(loader, desc="Ensemble", total=len(loader)):
                x=x.to(detector.device)
                votes=[m(x).argmax(1).cpu() for m in ensemble]
                votes=torch.stack(votes)
                mode=torch.mode(votes,0).values
                ens_dis.extend((votes!=mode).float().mean(0).numpy())
        risk = conf + inc + np.asarray(ens_dis)
    else:
        risk = conf + inc

    suspects = np.argsort(-risk)[:top_k]   # descending risk
    return suspects.tolist()


# ❶  run verbose detector
with torch.no_grad():
    suspects = rank_suspects_verbose(
        detector       = trainer.detector,
        loader         = trainer.train_loader,
        top_k          = 1000,
        n_aug          = 4,
        ensemble       = None              # or a list of extra models
    )

print("\nTop-10 suspect indices:", suspects[:10])

# ❷  tidy up GPU  (same as before)
for m in trainer.model.modules():
    if hasattr(m, "saved_attn"):
        delattr(m, "saved_attn")
del trainer.detector
gc.collect()
torch.cuda.empty_cache()
# ---------------------------------------------------------------------------

# ─── Visualise the 10 highest-risk samples (fixed) ──────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

suspects = [49882, 10663, 21799, 32211, 30263,
            19666, 36593, 32619, 21371, 35196]

class_names = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]   # CIFAR-10

ds    = trainer.train_dataset
model = trainer.model.eval()

prep = T.Compose([
    T.ToPILImage(),                                     # expects *tensor* C×H×W
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for ax, idx in zip(axs.flat, suspects):
    # raw CHW uint8 → tensor so ToPILImage sees C×H×W, not H×W×C
    img_chw = torch.from_numpy(ds.data[idx])            # tensor, uint8, 3×32×32
    x = prep(img_chw).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        conf, pred = torch.softmax(logits, 1).max(1)
    pred, conf = pred.item(), conf.item()

    # show original 32×32 RGB
    ax.imshow(ds.data[idx].transpose(1, 2, 0))
    ax.set_title(f"GT:  {class_names[int(ds.labels[idx])]}\n"
                 f"Pred: {class_names[pred]} ({conf:.2f})",
                 fontsize=9)
    ax.axis('off')

plt.tight_layout(); plt.show()



# Force training from scratch (ignoring any saved model/weights)
trainer = ViTTrainer(dataset="mnist", noise_ratio=0.3, batch_size=64, resume=False)

# Train the model for 5 epochs (the history and model weights will be saved)
history = trainer.train(num_epochs=5)

# Continue training for additional epochs
#history = trainer.continue_training(additional_epochs=2)

# Plot training and evaluation metrics
trainer.plot_metrics()



trainer2.confusion_matrix()

path ='/content/drive/MyDrive/Spring2025Capstone-HanDeepLearning/vit_new_models'
# Resume training from previously saved model weights and history.
trainer2 = ViTTrainer(dataset="cifar100", noise_ratio=0.3, batch_size=64, resume=True)

# Plot saved data
#trainer2.plot_metrics()
#trainer2.confusion_matrix()
#trainer2.visualize_attention(dataset_index=0, head_idx=0)

# Or continue training for additional epochs:
#history = trainer.continue_training(additional_epochs=5)

# Plot updated data after resuming training
trainer2.plot_metrics()

trainer2.visualize_attention(dataset_index=0, head_idx=0)

trainer = ViTTrainer(dataset="cifar10", noise_ratio=0.6, resume=True,
                     mixup=True, mixup_alpha=1.0)
history = trainer.train(num_epochs=5)  # Mixup is applied in the training loop

# Plot training and evaluation metrics
trainer.plot_metrics()