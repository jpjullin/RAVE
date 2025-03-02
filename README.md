![rave_logo](docs/rave.png)

## 🚀 Installation

1. Install `ffmpeg` (needs `Chocolatey`) from **Admin Powershell**
```bash
choco install ffmpeg -y
```

2. Activate virtualenv, then install `torch` & `torchaudio`
Find the right command on the [PyTorch website](https://pytorch.org/).


3. Install **RAVE**
```bash
pip install -r requirements.txt
```

## 🎛️ Usage

Training a RAVE model involves three steps: **dataset preparation, training, and export**.

### 📂 Dataset Preparation  

1. Prepare your audio dataset in a folder.
   - Use **at least 1-3 hour of high-quality audio**.  
   - **Balanced diversity** is key: too much variation makes training harder, too little limits generalization.  
   - Normalize or compress audio if loudness varies significantly.


2. Resample (if needed)
If your dataset contains **multiple sample rates**, first `cd` into your audio folder and run:
```bash
cd /path/to/audio/folder
resample --sr TARGET_SAMPLING_RATE --augment
```
⚠️ This will convert files to **16-bit WAV, mono, 44.1 kHz**.


3. Preprocess your dataset
Once your audio is ready, preprocess it:
```bash
python ./scripts/preprocess.py --input_path /audio/folder --output_path /dataset/path --channels X (--lazy)
```

- Add `--lazy` for very large datasets.
- Use `--channels 1` to use in RAVE VST.


### 🎯 Training

RAVEv2 supports multiple configurations. To train the v2 model, run:
```bash
python ./scripts/train.py --config v2 --db_path /dataset/path --out_path /model/out --name give_a_name --channels X
```

#### 🎨 Model Architectures
- **v1**: Original continuous model.
- **v2**: Improved continuous model (faster, higher quality, 16GB GPU)
- **v2_small**: Optimized for timbre transfer (8GB GPU)
- **v2_nopqmf**: v2 without pqmf in generator (experimental, for bending purposes, 16GB GPU)
- **v3**: Real style transfer with Snake activation (32GB GPU)
- **discrete**: Discrete model (similar to SoundStream/EnCodec, 18GB GPU)
- **onnx**: Optimized for ONNX export (6GB GPU)
- **raspberry**: Lightweight model for Raspberry Pi 4 (5GB GPU)

#### 🛠️ Regularization (v2 only)
- **default**: Use this one before any further experiment!
- **wasserstein**: Better reconstruction results, at the price of a more messy latent space.
- **spherical**: Enforces the latent space to be distributed on a sphere. It is experimental, do not try that first!

#### 🎛️ Discriminator
- **spectral_discriminator**: Use the MultiScale discriminator from EnCodec.

#### 🎭 Others
- **causal**: Reduces the perceived latency of the model, at the price of a lower reconstruction quality.
- **noise**: Better for learning sounds with important noisy components.
- **hybrid**: Enable mel-spectrogram input, may be useful for learning on voice.

Configurations can be combined:
```bash
python ./scripts/train.py --config ./rave/configs/v2 --config ./rave/configs/causal ...
```

#### 🎨 Augmentations (Linux only)
- **mute**: Randomly mutes data batches (default prob: 0.1).
- **compress**: Randomly compresses the waveform.
- **gain**: Applies a random gain to the waveform (default range: [-6, 3]).

Augmentations can be combined:
```bash
python ./scripts/train.py --augment ./rave/configs/augmentations/mute --augment ./rave/configs/augmentations/compress --augment ./rave/configs/augmentations/gain
```

To train on GPU, add `--gpu` and specify the GPU index with `--gpu_id X`.

### 🚀 Export

Once trained, export your model to a torchscript file using

```bash
python ./scripts/export.py --run /path/to/model/ --streaming
```
