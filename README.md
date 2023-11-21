# CTX-text2vec, the Acoustic Model with Contextual VQ-Diffusion
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
The decoding of CTX-text2vec always rely on prompts that provide contextual information. In other words, before decoding, there should be a `utt2prompt` file that looks like:
```text
1089_134686_000002_000001 1089_134686_000032_000008
1089_134686_000007_000005 1089_134686_000032_000008
1089_134686_000009_000003 1089_134686_000032_000008
1089_134686_000009_000008 1089_134686_000032_000008
1089_134686_000015_000003 1089_134686_000032_000008
```
where every line is organized as `utt-to-synthesize prompt-utt`.
The `utt-to-synthesize` and `prompt-utt` keys should both be present in `feats.scp` for indexing.

💡 We recommend using the [official utt2prompt file](https://cpdu.github.io/unicats/resources/testsetB_utt2prompt) for test set B in the paper.
You can download that and save to `data/eval_all/utt2prompt`.

After that, decoding with context prepended (a.k.a. continuation) can be performed by
```shell
python continuation.py --eval-set eval_all
# will only synthesize utterances in `utt2prompt`. Check the necessary files in `data/${eval_set}`.
```
The decoded VQ-indexes (2-dim) will be saved to `OUTPUT/Libritts/syn/${eval_set}/`.

> 💡Note that the model actually samples from 23631 distinct VQ "labels". In this code we transform it back to 2-dim VQ indexes using `feats/vqidx/label2vqidx`.

## Vocoding to waveform
For vocoding to waveform, the counterpart "[CTX-vec2wav](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav)" is highly recommended.
You can set up CTX-vec2wav by
```shell
git clone https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav.git
```
and then follow the environmental instruction there.

After decoding to VQ indexes, vocoding can be achieved by
```shell
syn_dir=$PWD/OUTPUT/Libritts/syn/eval_all/
utt2prompt_file=$PWD/data/eval_all/utt2prompt
v2w_dir=/path/to/CTX-vec2wav/

cd $v2w_dir || exit 1;
source path.sh
# now, in CTX-vec2wav's environment

feat-to-len.py scp:$syn_dir/feats.scp > $syn_dir/utt2num_frames
# construct acoustic prompt specifier (mel spectrograms) using utt2prompt
python ./local/get_prompt_scp.py feats/normed_fbank/eval_all/feats.scp ${utt2prompt_file} > $syn_dir/prompt.scp

decode.py --feats-scp $syn_dir/feats.scp \
          --prompt-scp $syn_dir/prompt.scp \
          --num-frames $syn_dir/utt2num_frames \
          --outdir $syn_dir/wav/ \
          --checkpoint /path/to/checkpoint
```

## Acknowledgement
During the development, the following repositories were referred to:
* [ESPnet](https://github.com/espnet/espnet), for the model architecture in `ctx_text2vec/modeling/transformers/espnet_nets` and utility scripts in `utils`.
* [Kaldi](https://github.com/kaldi-asr/kaldi), for most utility scripts in `utils`.
* [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), from which the model structures and training pipeline are mostly inherited.
* [CTX-vec2wav](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav) for vocoding!