# Versions
numpy                     1.19.5
scipy                     1.5.3
scikit-learn              0.24.0
cudatoolkit               11.3.1
pytorch                   1.10.2
tensorflow-gpu            1.14.0


# 0. Data preparation
## Data format
The input data is a dictionary with the following format:
'sample-id': [(tensor for amino acid sequence), (tensor for average BLOSUM embedding of the peptide), (label)]
sample-id format: {cdr3 sequence}-{peptide sequence}_{other info}

## Sample script
python 0.preprocess_data.py sample_train.tsv sample_train.pkl 

# 1. Model training
## Parameters
ep                     # Training epochs
per_gpu_batch_size     # per-gpu batch size
i_cls_weight           # Weight for the auxiliary classifier loss (L_f_cls)
recon_weight           # Weight for the main decoder reconstruction loss (L_recon)
s_recon_weight         # Weight for the structural embedding-based decoder reconstruction loss (L_s_recon) (not used in the reported results)
wass_weight            # Weight for the full reconstruction loss (L_wass)
sigma_rbf              # Sigma parameter for the radial basis function (RBF) kernel for the MMD approximation
epsilon                # Scheduled sampling probability (for decoder training)
data_path              # The path where the training data is stored
data_prefix            # Prefix of the data files (NOTE: please provide a prefix)
dataset                # Dataset prefix (not used)
output_path            # The path to store the model output
seed                   # The random seed
peptide                # The peptide to optimize on (in the training phase, it is not specified)

## Input files
The datasets for model training should be dumped into .pkl objects with a common data_prefix followed by train/test/val. Provide a .tsv file containing the cdr3-peptide pairs to 0.preprocess_data.py.

${data_prefix}_train.pkl
${data_prefix}_test.pkl
${data_prefix}_val.pkl


## Sample script
python 1.train.py \
    --epochs ${ep} \
    --eval_step 500 \
    --per_gpu_batch_size ${batch_size} \
    --lr ${lr} \
    --i_cls_weight ${cls_weight}\
    --recon_weight ${recon_weight}\
    --s_recon_weight ${s_recon_weight}\
    --wass_weight ${wass_weight}\
    --sigma_rbf ${sigma_rbf}\
    --epsilon ${epsilon}\
    --data_path ${data_path}\
    --dataset ${dset}\
    --data_prefix ${data_prefix}\
    --save_model_path ${output_path}\
    --seed ${seed} \
    --peptide ${peptide}


# 2. TCR optimization
##Parameters
batch_size             # per-gpu batch size
data_file              # Data file name (NOTE: please provide a full file name)
load_model_path        # The path to the trained model
dataset                # Dataset prefix (used for naming results)
output_path            # The path to store optimized sequences
output_result_prefix   # The prefix of the optimized sequence files
seed                   # The random seed
peptide                # The peptide to optimize on (REQUIRED)

## Input files
The datasets for model training should be dumped into .pkl objects with any file name. Provide a .tsv file containing the cdr3-peptide pairs to 0.preprocess_data.py.

## Sample script
python 2.train_optimize_external.py \
    --per_gpu_batch_size ${batch_size} \
    --dataset ${dset}\
    --data_file ${data_file}\
    --load_model_path \${load_path}\
    --load_model_prefix ${pt_prefix}\
    --output_path ${output_path}\
    --output_result_prefix ${pt_prefix}\
    --peptide ${peptide}\
