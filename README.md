# T2E: Tracing Transaction Evidence for Training-Free Temporal Knowledge Graph Forecasting

## Requirements

Tested on Linux with Python 3.8, PyTorch 2.4.1 and CUDA 12.1.

```bash
conda create -n t2e python=3.8 -y
conda activate t2e
pip install -r requirements.txt
```

## Model

Download the base [InternLM2-7B](https://huggingface.co/internlm/internlm2-7b) model:

Replace `/path/to/internlm2-7b` with your local model directory in both the download and prediction commands.

## Prediction

The datasets are included in `data/`. Run the following commands from the repository root.
Each command performs evidence retrieval, prediction and evaluation automatically.

### ICEWS14

```bash
python run.py --dataset ICEWS14 --model-path /path/to/internlm2-7b \
  --window 7 --tau 0.75 --packet-cap unlimited --symbolic-weight 0.3
```

### ICEWS05-15

```bash
python run.py --dataset ICEWS05-15 --model-path /path/to/internlm2-7b \
  --window 7 --tau 0.5 --packet-cap unlimited --symbolic-weight 0.7
```

Predictions and metrics are saved in `outputs/<dataset>-<timestamp>/`.

## Acknowledgements

Our implementation builds on [ONSEP](https://github.com/aqSeabiscuit/ONSEP) and [ICL](https://github.com/usc-isi-i2/isi-tkg-icl). We thank the authors for sharing their code.
