# \[Working in Progress\] CTX-text2vec, the Acoustic Model with Contextual VQ-Diffusion
> This is the official implementation of **CTX-text2vec** TTS model in the paper [UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547).

![main](asset/main.png)

## Environment Setup

This repo is tested on **python 3.7** on Linux. You can set up the environment with conda
```shell
# Install required packages
conda create -n ctxt2v python=3.7 # or any name you like
conda activate ctxt2v
pip install -r requirements.txt
```

Every time you enter this project, you can do `conda activate ctxt2v` or `source path.sh`.

Also, you can perform `chmod +x utils/*` to ensure those scipts are executable.

## Data Preparation
Here we take the LibriTTS preparation pipeline for example. Other datasets can be set up in the same way.

1. Please download the data manifests from [huggingface (38MB)](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/data_ctxt2v.zip).
    Then, unzip it to `data/` in the project directory. The contents are as follows:
    ```
    ├── train_all
    │         ├── duration    # the integer duration for each utterance. Frame shift is 10ms.
    │         ├── feats.scp   # the VQ index for each utterance. Will be explained later.
    │         ├── text   # the phone sequence for each utterance
    │         └── utt2num_frames   # the number of frames of each utterance.
    ├── eval_all
    │         ...  # similar four files
    │── dev_all
    │         ...
    └── lang_1phn
              └── train_all_units.txt  # mapping between valid phones and their indexes
    ```
2. Here, the `feats.scp` is the Kaldi-style feature specifier pointing to `feats/label/.../feats.ark`.
   We also provide it [online (432MB)](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/feats_ctxt2v.zip), so please download it and unzip to `feats` in the project directory.
   These features are the 1-D flatten indexes of the vq-wav2vec features. You can verify the shape of features by `utils/feat-to-shape.py scp:feats/label/dev_all/feats.scp | head`. The codebook `feats/vqidx/codebook.npy` has shape `[2, 320, 256]`.
> 💡 That is, we extracted discrete codebook indxes using [fairseq's vq-wav2vec model](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#vq-wav2vec) which contained 2 groups of integer indexes each ranging from 0 to 319.
   We then find the occurrences of these pairs and label them using another index, which counts to 23632. The mapping between this label index and original vq-wav2vec codebook index can be found at `feats/vqidx/label2vqidx`. We use the 23632 labels to train the VQ-diffusion model.

After constructing the directories properly, the model can be trained.

## Training

Training the CTX-text2vec model can be simply done by

```shell
python train.py --name Libritts --config_file configs/Libritts.yaml --num_node 1 --tensorboard --auto_resume
```
where `--name` specifies the output directory name. Check out `configs/Libritts.yaml` for detailed configurations.
After the training starts, checkpoints and logs will be saved in `OUTPUT/Libritts`.

## Decoding to VQ indexes
**Working in progress**

## Vocoding to waveform
**Working in progress**

```shell
python continuation.py
```

## Acknowledgement
During the development, the following repositories were referred to:
* [ESPnet](https://github.com/espnet/espnet), for the model architecture in `ctx_text2vec/modeling/transformers/espnet_nets` and utility scripts in `utils`.
* [Kaldi](https://github.com/kaldi-asr/kaldi), for most utility scripts in `utils`.
* [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), from which the model structures and training pipeline are mostly inherited.