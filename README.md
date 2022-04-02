# MTGAN

Code for the paper: *Multi-Label Clinical Time-Series Generation via Conditional GAN*

## Commands

### Preprocess Data

1. Go to https://mimic.physionet.org/ for access. Once you have the authority for the dataset, download the dataset and extract the csv files to {data_path}/mimic3/raw/ and {data_path}/mimic4/raw/.
2. Check hyper-parameter settings:
```bash
python run_preprocess.py --help
```
3. Run preprocess program:

- For MIMIC-III
```bash
python run_preprocess.py --dataset mimic3 --train_num 6000
```
- For MIMIC-IV
```bash
python run_preprocess.py --dataset mimic4 --train_num 6000 --sample_num 10000
```

Add `--from_saved` if you run it again and want to load saved encoded data.

### Training MTGAN
1. Check hyper-parameter setting:
```bash
python run_train.py --help
```

2. Run training program
- For MIMIC-III
```bash
python run_train.py --dataset mimic3
```
- For MIMIC-IV
 ```bash
python run_train.py --dataset mimic4
```

3. The parameters are saved in default at `results/{dataset}/params/`. The training plots are saved at `results/{dataset}/records/`.

### Generating EHR data
1. Check hyper-parameter setting:
```bash
python run_generate.py --help
```

2. Run generating program
- For MIMIC-III
```bash
python run_generate.py --dataset mimic3
```
- For MIMIC-IV
 ```bash
python run_generate.py --dataset mimic4
```

3. The generated data are saved at `results/synthetic_{dataset}.npz`.

## Requirements:

### Environment

- Python >= 3.7
- Virtualenv (optional, recommended)
- CUDA (optional, recommended)
- RAM > 16GB

### Installing packages:

- matplotlib
- numpy
- openpyxl
- pandas
- scipy
- pytorch
