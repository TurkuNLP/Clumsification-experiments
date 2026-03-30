from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn



class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # constrains to [0, 1]

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)
    
def load_model_and_head(
        model_path: str,
        head_path: str,
        emb_dim: int = 768,
    ):
        #Reload the encoder + scoring head from disk.
        loaded_model = SentenceTransformer(model_path)

        loaded_head = nn.Linear(emb_dim, 1)
        loaded_head.load_state_dict(torch.load(head_path))
        loaded_head.eval()

        return loaded_model, loaded_head
    
def main():

    #Add to this
    MODEL_PATH = ""
    HEAD_PATH = ""

    model, score_head = load_model_and_head(
         MODEL_PATH,
         HEAD_PATH
    )

    score_head.eval()

    # Inference
    with torch.no_grad():
        embeddings = model.encode(test_texts, convert_to_tensor=True)
        raw_preds = score_head(embeddings).numpy()

    # Spearman works directly — no rescaling required
    spearman, _ = spearmanr(test_labels, raw_preds)
    print(f"Spearman ρ: {spearman:.4f}")

    # 3. Rescale: [0, 1] -> [1, 5]
    def rescale(preds, source_min=0, source_max=1, target_min=1, target_max=5):
        return target_min + (preds - source_min) / (source_max - source_min) * (target_max - target_min)

    scaled_preds = rescale(np.array(raw_preds))

    # 4. Evaluate
    test_labels_np = np.array(test_labels)

    spearman, _ = spearmanr(test_labels_np, scaled_preds)
    pearson, _  = pearsonr(test_labels_np, scaled_preds)
    mae         = mean_absolute_error(test_labels_np, scaled_preds)
    mse         = mean_squared_error(test_labels_np, scaled_preds)

    print(f"Spearman: {spearman:.4f}")
    print(f"Pearson:  {pearson:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"MSE:      {mse:.4f}")