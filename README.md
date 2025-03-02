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

#### 🎨 Augmentations
- **mute**: Randomly mutes data batches (default prob: 0.1).
- **compress**: Randomly compresses the waveform.
- **gain**: Applies a random gain to the waveform (default range: [-6, 3]).

Configurations can be combined:
```bash
python ./scripts/train.py --config v2 --config consal --augment compress...
```



### Export

Once trained, export your model to a torchscript file using

```bash
rave export --run /path/to/your/run (--streaming)
```

Setting the `--streaming` flag will enable cached convolutions, making the model compatible with realtime processing. **If you forget to use the streaming mode and try to load the model in Max, you will hear clicking artifacts.**

## Prior

For discrete models, we redirect the user to the `msprior` library [here](https://github.com/caillonantoine/msprior). However, as this library is still experimental, the prior from version 1.x has been re-integrated in v2.3.

### Training

To train a prior for a pretrained RAVE model :

```bash
rave train_prior --model /path/to/your/run --db_path /path/to/your_preprocessed_data --out_path /path/to/output
```

this will train a prior over the latent of the pretrained model `path/to/your/run`, and save the model and tensorboard logs to folder `/path/to/output`.

### Scripting

To script a prior along with a RAVE model, export your model by providing the `--prior` keyword to your pretrained prior :

```bash
rave export --run /path/to/your/run --prior /path/to/your/prior (--streaming)
```

## Pretrained models

Several pretrained streaming models [are available here](https://acids-ircam.github.io/rave_models_download). We'll keep the list updated with new models.

## Realtime usage

This section presents how RAVE can be loaded inside [`nn~`](https://acids-ircam.github.io/nn_tilde/) in order to be used live with Max/MSP or PureData.

### Reconstruction

A pretrained RAVE model named `darbouka.gin` available on your computer can be loaded inside `nn~` using the following syntax, where the default method is set to forward (i.e. encode then decode)

<img src="docs/rave_method_forward.png" width=400px/>

This does the same thing as the following patch, but slightly faster.

<img src="docs/rave_encode_decode.png" width=210px />

### High-level manipulation

Having an explicit access to the latent representation yielded by RAVE allows us to interact with the representation using Max/MSP or PureData signal processing tools:

<img src="docs/rave_high_level.png" width=310px />

### Style transfer

By default, RAVE can be used as a style transfer tool, based on the large compression ratio of the model. We recently added a technique inspired from StyleGAN to include Adaptive Instance Normalization to the reconstruction process, effectively allowing to define _source_ and _target_ styles directly inside Max/MSP or PureData, using the attribute system of `nn~`.

<img src="docs/rave_attribute.png" width=550px>

Other attributes, such as `enable` or `gpu` can enable/disable computation, or use the gpu to speed up things (still experimental).

## Offline usage

A batch generation script has been released in v2.3 to allow transformation of large amount of files

```bash
rave generate model_path path_1 path_2 --out out_path
```

where `model_path` is the path to your trained model (original or scripted), `path_X` a list of audio files or directories, and `out_path` the out directory of the generations.

## Discussion

If you have questions, want to share your experience with RAVE or share musical pieces done with the model, you can use the [Discussion tab](https://github.com/acids-ircam/RAVE/discussions) !

## Demonstration

### RAVE x nn~

Demonstration of what you can do with RAVE and the nn~ external for maxmsp !

[![RAVE x nn~](http://img.youtube.com/vi/dMZs04TzxUI/mqdefault.jpg)](https://www.youtube.com/watch?v=dMZs04TzxUI)

### embedded RAVE

Using nn~ for puredata, RAVE can be used in realtime on embedded platforms !

[![RAVE x nn~](http://img.youtube.com/vi/jAIRf4nGgYI/mqdefault.jpg)](https://www.youtube.com/watch?v=jAIRf4nGgYI)

# Frequently Asked Question (FAQ)

**Question** : my preprocessing is stuck, showing `0it[00:00, ?it/s]`<br/>
**Answer** : This means that the audio files in your dataset are too short to provide a sufficient temporal scope to RAVE. Try decreasing the signal window with the `--num_signal XXX(samples)` with `preprocess`, without forgetting afterwards to add the `--n_signal XXX(samples)` with `train`

**Question** : During training I got an exception resembling `ValueError: n_components=128 must be between 0 and min(n_samples, n_features)=64 with svd_solver='full'`<br/>
**Answer** : This means that your dataset does not have enough data batches to compute the intern latent PCA, that requires at least 128 examples (then batches). 


# Funding

This work is led at IRCAM, and has been funded by the following projects

- [ANR MakiMono](https://acids.ircam.fr/course/makimono/)
- [ACTOR](https://www.actorproject.org/)
- [DAFNE+](https://dafneplus.eu/) N° 101061548

<img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_co_funded_en.jpg" width=200px/>
