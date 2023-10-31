train_set=train_all
dev_set=dev_all
eval_set=eval_all
featdir=feats
nj=8
. path.sh

vqdir=${featdir}/vqidx
./local/get_label2vqidx_dict.py <( cat ${vqdir}/{${train_set},${dev_set},${eval_set}}/feats.scp ) > ${vqdir}/label2vqidx
for x in ${train_set} ${dev_set} ${eval_set} ; do
    mkdir -p ${featdir}/label/${x}
    ./local/vqidx2label.py ${vqdir}/label2vqidx scp:${vqdir}/${x}/feats.scp ark,scp:${featdir}/label/${x}/feats.ark,${featdir}/label/${x}/feats.scp
    cp ${featdir}/label/${x}/feats.scp data/${x}/feats.scp
done

source activate ~/tools/anaconda3/envs/vqdiffusion
python train.py --name Libritts --config_file configs/Libritts.yaml --num_node 1 --tensorboard --auto_resume

python continuation.py
# a=`ls OUTPUT/Libritts_thres24/syn/train_clean/npy/ | wc -l` ; while [ $a -lt 500  ] ; do python continuation.py ; a=`ls OUTPUT/Libritts_thres24/syn/train_clean/npy/ | wc -l` ;  done

eval_dir=$PWD/OUTPUT/Libritts/syn/${eval_set}
voc_dir=/mnt/lustre/sjtu/home/cpd30/tools/ParallelWaveGAN/egs/libritts/v2w2

awk 'NR==FNR{a[$1]=$2}NR>FNR{printf("%s %s\n", $1, a[$2])}' feats/normed_fbank/${eval_set}/feats.scp data/${eval_set}/utt2prompt_subset > ${eval_dir}/prompt.scp
cd ${voc_dir} || exit
bash local/generate.sh --conf conf/hifigan.v1.yaml --eval_dir ${eval_dir}
