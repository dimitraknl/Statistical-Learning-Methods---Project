TITLE: Project for Modern Methods to statistical learning 


The dataset used for this project can be downloaded from Kaggle:

> https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset

After downloading, place the extracted folder in the project root directory and rename it as:

Alzheimer_Dataset_V2/

Project folder structure:
ml-project/
│
├── src/ # All source code
│ ├── train_cnn.py # CNN model training script
│ ├── train_svm_grid.py # SVM model training (grid search for Linear, RBF, Polynomial)
│ ├── evaluate_model.py # Evaluation script (CNN and SVM)
│ └── plots.py # Visualization and bias–variance analysis
│
├── models/ # Saved trained/best models
│ ├── cnn_best.pth
│ ├── svm_rbf.pkl
│ ├── svm_linear.pkl
│ └── svm_poly.pkl
│
├── results/ # Training and evaluation outputs
│ ├── cnn/
│ │ ├── training_logs/
│ │ └── evaluation_reports/
│ │
│ └── svm_grid/
│ ├── Linear/
│ ├── Polynomial/
│ └── RBF/

Environment setup:
pip install -r requirements.txt


Run the following in your terminal:

CNN Training:
python src/train_cnn.py \
  --data-root Alzheimer_Dataset_V2 \
  --epochs 4 \
  --batch-size 32 \
  --out models/cnn_best.pth

Evaluation:
python src/evaluate_model.py \
  --model-type cnn \
  --model-path models/cnn_best.pth \
  --data-root Alzheimer_Dataset_V2 \
  --split test


CNN Training:
python src/train_svm_grid.py \
  --kernels rbf,linear,poly \
  --Cs 0.1,1,10 \
  --gammas scale,0.001,0.01 \
  --degrees 2,3 

See parse arguments for additional commands (optional)

Evaluation (RBF):
python src/evaluate_model.py \
  --model-type svm \
  --model-path models/svm_rbf.pkl \
  --data-root Alzheimer_Dataset_V2 \
  --split test

Evaluation (Linear):
python src/evaluate_model.py \
  --model-type svm \
  --model-path models/svm_linear.pkl \
  --data-root Alzheimer_Dataset_V2 \
  --split test

Evaluation (Polynomial):
python src/evaluate_model.py \
  --model-type svm \
  --model-path models/svm_poly.pkl \
  --data-root Alzheimer_Dataset_V2 \
  --split test


PLOTS: 
python src/plots.py --kernel RBF\Linear\Polynomial

Results
CNN: results under results/cnn/
SVM: results for each kernel under results/svm_grid/Linear, Polynomial, and RBF
Metrics include accuracy, F1-score, confusion matrices, and per-class performance visualizations.
