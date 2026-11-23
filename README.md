# IRRA + hard-candidate contrastive branch

This experiment keeps IRRA as the base text-to-image retriever and adds a
trainable second contrastive objective over the small set of candidates that
IRRA considers hardest for each query. It is intended as a first 10-epoch
feasibility run on RSTPReid.

## Architecture

IRRA uses its CLS image feature and EOT text feature to retrieve the initial
top-K images. For each training query, the branch then:

1. Uses the top-K images from the current minibatch as hard candidates.
2. Checks whether the top-K already contains an image with the query PID. If
   not, it appends exactly one positive image from the minibatch. If a positive
   is already in top-K, it adds no extra images.
3. Maps the query and candidate CLS features into a lower-dimensional learned
   space with separate text and image projection heads.
4. Applies supervised contrastive loss to the query and the selected images.
   Features with the same identity are pulled together; features with different
   identities are pushed apart. This is the second contrastive objective, in
   addition to IRRA's original objectives.

The projection is learned jointly with IRRA. It is not a fixed PCA transform:
PCA preserves directions with high variance but does not know which pairs share
an identity. The identity-supervised contrastive objective teaches the
projection which distinctions matter for retrieval.

The total loss is

\[
L = L_{SDM} + L_{ID} + L_{MLM} + \lambda_h L_{hard\_contrastive}.
\]

At inference, IRRA ranks the complete gallery and selects top-K candidates. The
projection branch adjusts the scores of those candidates using query-image
similarity in the learned space. Images outside top-K keep their original IRRA
scores. No identity labels are used at inference. Only top-K scores are
adjusted; the rest of the gallery remains eligible with its original IRRA
scores, but the branch does not directly promote those outside candidates.

The training candidate pool is the minibatch gallery, while inference uses the
full gallery. This is a batch-hard approximation and should be compared against
the IRRA baseline. This branch does not use the previous graph-convolution or
Transformer-memory modules.

## Data

The Kaggle dataset is expected to contain:

```text
benchmark/
└── RSTPReid/
    ├── imgs/
    └── data_captions.json
```

Set `root_dir` to the `benchmark` directory, the parent of `RSTPReid`.

## Environment

Use Python 3.10. Install PyTorch and TorchVision for the CUDA version on the
training machine, then install the remaining requirements:

```bash
conda create -n irra-hard python=3.10 -y
conda activate irra-hard
# Install torch and torchvision for the machine's CUDA version.
pip install -r requirements.txt
python scripts/prepare_tokenizer.py
```

The first run also downloads the CLIP ViT-B/16 weights. On Kaggle, enable
Internet for the BPE tokenizer and CLIP weight downloads.

## Train for 10 epochs

This run follows IRRA's published training recipe: random sampling, batch size
64, Adam at learning rate `1e-5`, image augmentation, and the `sdm+mlm+id`
losses. It runs for 10 epochs as a smoke-scale experiment while keeping the
learning-rate schedule on IRRA's 60-epoch horizon. The added projection
dimension is 128 and its contrastive temperature is 0.07.

```bash
python train.py \
  --name rstp_hard_contrastive_10ep \
  --dataset_name RSTPReid \
  --root_dir /path/to/tbps-benchmark/benchmark \
  --num_epoch 10 \
  --lr-total-epochs 60 \
  --optimizer Adam \
  --lr 1e-5 \
  --lr_factor 5 \
  --weight_decay 4e-5 \
  --batch_size 64 \
  --sampler random \
  --num_workers 8 \
  --img_aug \
  --MLM \
  --loss_names sdm+mlm+id \
  --candidate_topk 8 \
  --hard_projection_dim 128 \
  --hard_temperature 0.07 \
  --hard_score_weight 0.5
```

Validation runs each epoch and reports both `t2i-base` and
`t2i-hard-contrastive` R@1/5/10, mAP, and mINP. The best checkpoint is
selected by the gallery-contrastive validation R@1. Run test evaluation only
after model selection:

```bash
python test.py --config_file /path/to/run/configs.yaml --split test
```

## Run on Kaggle

Attach `hoanggv/tbps-benchmark` as the notebook input. In notebook Settings,
choose **Accelerator → GPU** and turn on **Internet**. The available GPU and
quota depend on Kaggle account eligibility and capacity. The example below
uses the dataset path from the supplied notebook.

**Cell 1 — clone this branch and prepare dependencies:**

```python
%cd /kaggle/working
!git clone --depth 1 -b codex/hard-candidate-contrastive https://github.com/lqb464/irra-gallery.git
%cd /kaggle/working/irra-gallery
!pip install -q -r requirements.txt
!python scripts/prepare_tokenizer.py
```

**Cell 2 — train for 10 epochs:**

```python
!python train.py \
  --name rstp_hard_contrastive_10ep \
  --dataset_name RSTPReid \
  --root_dir /kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark \
  --output_dir /kaggle/working/irra-hard-outputs \
  --num_epoch 10 \
  --lr-total-epochs 60 \
  --optimizer Adam \
  --lr 1e-5 \
  --lr_factor 5 \
  --weight_decay 4e-5 \
  --batch_size 64 \
  --sampler random \
  --num_workers 8 \
  --img_aug \
  --MLM \
  --loss_names sdm+mlm+id \
  --candidate_topk 8 \
  --hard_projection_dim 128 \
  --hard_temperature 0.07 \
  --hard_score_weight 0.5 \
  --val_dataset val
```

The dataset under `/kaggle/input` is read-only. Logs and checkpoints are
written under `/kaggle/working/irra-hard-outputs`. Save a Notebook Version to
retain outputs after the session ends.

## Ablations

- Run the IRRA baseline with `--disable_gallery_context`.
- Compare `--candidate_topk` values such as 4, 8, and 16.
- Compare projection dimensions such as 64, 128, and 256.
- Tune `--hard_score_weight` on validation. For example:

```bash
python test.py \
  --config_file /path/to/run/configs.yaml \
  --split val \
  --hard_score_weight 0.25
```

Choose settings on validation, then evaluate the selected checkpoint on test.
