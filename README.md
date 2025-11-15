## Berlin X-Ray Federated Classification (Hackathon Submission)

This repository contains our hackathon solution for federated chest X‑ray classification built with **Flower** and **PyTorch**.  
The goal is to collaboratively train a model across multiple hospitals without ever sharing raw images.

At a high level:
- Each hospital trains locally on its own chest X‑ray data.
- A central Flower server aggregates model updates with FedAvg.
- We track per‑hospital and global metrics and automatically save the best global model to Weights & Biases (W&B).

---

## 1. Approach in a Nutshell

- **Task**: Binary classification – predict **“any finding” vs “no finding”** on chest X‑rays.
- **Federated setup**: Three hospitals (`HospitalA`, `HospitalB`, `HospitalC`) act as Flower clients.  
  Only model weights and scalar metrics are communicated; no images leave the hospitals.
- **Backbone**: `DenseNet‑121` pretrained on ImageNet (`torchvision.models`).
- **Grayscale adaptation**: We replace the first convolution layer to accept **1‑channel** images and initialize it by averaging the original RGB weights.
- **Fine‑tuning strategy**:
  - Freeze early layers (generic features).
  - Unfreeze only the top DenseNet blocks and classifier.
  - Optimize with `AdamW` + `OneCycleLR` for fast, stable convergence per federated round.
- **Loss & outputs**:
  - Single logit output with `BCEWithLogitsLoss` for binary classification.
  - Evaluation uses ROC‑AUC and standard confusion‑matrix metrics (sensitivity, specificity, precision, F1).
- **Compute/time budget**:
  - Each full federated run was limited to about **20 minutes on a single AMD MI300X GPU**, so we favored partial fine‑tuning, a relatively large batch size (512), and an aggressive OneCycle schedule to reach strong performance within this tight time budget.

### 1.1. Model details

- **Architecture**:
  - `Net` wraps `torchvision.models.densenet121` with ImageNet weights.
  - We keep the DenseNet feature extractor and only change:
    - The first convolution (`conv0`) from 3 input channels to 1 (grayscale).
    - The final classifier to a single‑unit linear layer for binary classification.
  - Why DenseNet‑121: dense skip connections promote feature reuse and stable gradients, and this backbone is a well‑validated choice for chest X‑ray modeling, so we get strong performance without designing a custom architecture from scratch.
  - Why minimal surgery: changing only the input and output layers maximizes reuse of the pretrained representation while keeping the number of newly‑initialized parameters small.
- **Input handling**:
  - `Net._prepare_input` ensures inputs are `float32`, rescales values from `[-1, 1]` to `[0, 1]` if needed, and resizes every image to the chosen `image_size` using bilinear interpolation.
  - This makes the model robust to slightly different preprocessing pipelines and resolutions across hospitals while keeping the backbone expectations consistent.
- **Frozen vs trainable layers**:
  - All parameters are frozen by default, then only the following are unfrozen: `denseblock3`, `transition3`, `denseblock4`, `norm5`, and the final classifier.
  - Intuition: early convolutional layers capture generic edges/textures that transfer well across datasets; we adapt only higher‑level representations to chest X‑ray patterns and hospital‑specific distributions.
- **Optimization**:
  - We optimize with `AdamW` (weight decay `0.01`) on the unfrozen parameters only.
  - `OneCycleLR` schedules the learning rate over `local-epochs × steps_per_epoch`, with a short warm‑up and gradual decay, so each hospital can make meaningful progress in a few local epochs before synchronization.
  - Why AdamW: it is less sensitive to learning‑rate choice than plain SGD, handles different layer scales well (important when mixing frozen and unfrozen blocks), and the decoupled weight decay regularizes the high‑level layers we are adapting.
  - Why OneCycle: we only get a small number of local epochs per round; OneCycle is designed to extract as much performance as possible from a limited number of steps, avoiding long warm‑up or slow plateau phases.
- **Batch size**:
  - In our **winning configuration**, we used a **batch size of 512** per hospital, which strikes a balance between stable gradient estimates and efficient utilization of the shared GPU resources.
  - Why 512: larger batches reduce gradient noise and make the learning‑rate schedule more predictable, while still fitting comfortably into the available GPU memory on the hackathon infrastructure.
- **Predictions**:
  - During evaluation we apply a sigmoid to the logit to obtain a probability of “finding present”.
  - A default threshold of 0.5 is used to derive binary predictions, from which we compute sensitivity, specificity, precision, F1, and ROC‑AUC.

---

## 2. Repository Structure

- `berlin25_xray/`
  - `task.py` – Model definition (`Net`), data loading, training & evaluation logic, and reference preprocessing.
  - `client_app.py` – Flower **ClientApp**: local train/eval on each hospital’s data.
  - `server_app.py` – Flower **ServerApp**: FedAvg strategy, logging, best‑model saving.
  - `util.py` – Metric computation and W&B helpers (per‑hospital and global logging).
  - `__init__.py` – Package init.
- `evaluate.py` – Offline evaluation script for a saved W&B model artifact.
- `local_train.py` – Simple local (non‑federated) training loop for debugging on one hospital.
- `pyproject.toml` – Flower app configuration (server/client components, federations, default hyperparameters).
- `requirements.txt` – Environment setup for the shared hackathon cluster.

---

## 3. Data & Preprocessing

- We build on the hackathon organizers’ preprocessed chest X‑ray datasets, exposed as Hugging Face `datasets` objects.
- Each example is a pair `(x, y)` where:
  - `x` is a single‑channel image tensor already resized to the configured `image-size` (e.g. 224×224) and roughly normalized.
  - `y` is a binary label: `0` if **“No Finding”** is the only label, `1` if any other pathology label is present.
- Data is partitioned into per‑hospital splits (`HospitalA`, `HospitalB`, `HospitalC`) plus held‑out test sets, which directly matches the federated multi‑hospital scenario.
- The reference transformation pipeline used to go from raw images to `x` is implemented in `berlin25_xray/task.py:apply_transforms` and consists of:
  - Resize → Grayscale (1 channel) → ToTensor → Normalize(mean=[0.5], std=[0.5]).
- During training and evaluation we simply load `x` and `y` from disk and feed them into the model; all federated logic operates on these tensors, never on raw images.

---

## 4. Federated Training with Flower

The Flower app is configured in `pyproject.toml`:

- `serverapp = "berlin25_xray.server_app:app"`
- `clientapp = "berlin25_xray.client_app:app"`
- Default config:
  - `image-size = 224`
  - `num-server-rounds = 100`
  - `local-epochs = 2`
  - `lr = 0.001`

In the federated setup:
- All hospitals start from the same global DenseNet‑121 checkpoint.
- Each hospital runs a few local epochs of fine‑tuning on its own data with the optimization setup described above.
- The Flower server aggregates the resulting weights with FedAvg, weighting updates by data volume so that larger hospitals have proportionally more influence.
- After each round, we evaluate the updated global model on every hospital, compute ROC‑AUC and confusion‑matrix metrics, and keep track of the best‑performing global checkpoint.

---

## 5. What’s Special About This Solution?

- **Strong, robust backbone**: DenseNet‑121 with ImageNet pretraining, adapted for grayscale X‑rays.
- **Careful fine‑tuning**: Only higher layers are trainable, which:
  - Reduces overfitting.
  - Speeds up training under tight round and time budgets.
- **Federated by design**: Built around Flower’s ServerApp/ClientApp APIs with clear separation of client and server logic.
- **Rich metrics**:
  - Per‑hospital and global ROC‑AUC.
  - Clinically relevant metrics: sensitivity, specificity, precision, F1.
- **Experiment tracking & artifacts**:
  - All metrics logged to W&B.
  - Best global model automatically saved as a versioned W&B artifact.

This makes the solution both **practical for the hackathon setting** (easy to run on the provided infrastructure) and **realistic** for real‑world multi‑hospital collaboration.
