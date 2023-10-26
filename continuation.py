from tqdm import tqdm
import time
import numpy as np
import kaldiio
import torch

import os
from ctx_text2vec.utils.io import load_yaml_config
from ctx_text2vec.modeling.build import build_model

eval_set="eval_clean"
expdir = "OUTPUT/Libritts"

device = "cuda"
config = load_yaml_config(f'{expdir}/configs/config.yaml')
model = build_model(config).to(device)
ckpt = torch.load(f"{expdir}/checkpoint/last.pth")
outdir = f"{expdir}/syn/{eval_set}/npy"
model.load_state_dict(ckpt["model"])


lexicon = {}
with open("data/lang_1phn/train_all_units.txt", 'r') as f:
    for line in f.readlines():
        txt_token, token_id = line.strip().split()
        lexicon[txt_token] = int(token_id)

vqid_table = []
with open("feats/vqidx/label2vqidx", 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        label = int(line[0])
        vqid_table.append(torch.tensor(list(map(int, line[1:]))))
vqid_table = torch.stack(vqid_table, dim=0).to(device)

utt2text = {}
with open(f"data/{eval_set}/text") as f:
    for line in f.readlines():
        utt, text = line.strip().split(maxsplit=1)
        utt2text[utt] = text

utt2dur = {}
with open(f"data/{eval_set}/duration") as f:
    for line in f.readlines():
        utt, duration = line.strip().split(maxsplit=1)
        utt2dur[utt] = list(map(int, duration.split()))

feats_loader = kaldiio.load_scp(f'data/{eval_set}/feats.scp')

if not os.path.exists(outdir):
    os.makedirs(outdir)
feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=os.path.join(os.getcwd(), f"{outdir}/feats")))

with open(f"data/{eval_set}/utt2prompt_subset") as f:
    # last_time = time.time()
    model.set_generate_type('top0.85r')
    for l in tqdm(f.readlines()):
        utt, prompt = l.strip().split(maxsplit=1)
        # save_dir = f"{outdir}/{utt}.npy"
        # if os.path.exists(save_dir):
        #     continue
        text = utt2text[utt]
        text = torch.LongTensor([lexicon[w] for w in text.split()]).unsqueeze(0).to(device)
        prefix_text = torch.LongTensor([lexicon[w] for w in utt2text[prompt].split()]).unsqueeze(0).to(device)
        duration = torch.LongTensor(utt2dur[utt]).unsqueeze(0).to(device)
        prefix_duration = torch.LongTensor(utt2dur[prompt]).unsqueeze(0).to(device)
        feat = torch.LongTensor(feats_loader[utt][:, -1].copy()).unsqueeze(0).to(device)
        prefix_feat = torch.LongTensor(feats_loader[prompt][:, -1].copy()).unsqueeze(0).to(device)
        out = model.transformer.sample(text, {'text': prefix_text, 'duration': prefix_duration, 'feat': prefix_feat})['content_token'][0]
        out = out[prefix_feat.size(-1):]
        vqid = vqid_table[out].float().cpu().numpy()
        feat_writer[utt] = vqid
        # np.save(save_dir, vqid)
        # if time.time() - last_time > 60:
        #     break
        # last_time = time.time()

feat_writer.close()
