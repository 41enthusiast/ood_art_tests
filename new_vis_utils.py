import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import torch
from torch import nn
import torchvision
import timm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
from renumics import spotlight
from annoy import AnnoyIndex

from stclf_model import stclf_model
import pandas as pd
from sampler_utils_vis import *

txt_classes = {0: 'noble',
               1: 'warrior',
               2: 'incarnation',
               3: 'commoner'}


def visualize_outliers(idxs, data):
    data_subset = torch.utils.data.Subset(data, idxs)
    plot_images(data_subset)

def outlier_score_by_embeddings_cleanlab(df, embedding_name="embedding"):
    """
    Calculate outlier score by embeddings using cleanlab
        Args:
            df: dataframe with embeddings
            embedding_name: name of the column with embeddings
        Returns:
            new df_out: dataframe with outlier score
    """
    embs = np.stack(df[embedding_name].to_numpy())
    ood = OutOfDistribution()
    ood_train_feature_scores = ood.fit_score(features=np.stack(embs))
    df_out = pd.DataFrame()
    df_out["outlier_score_embedding"] = ood_train_feature_scores
    return df_out


def extract_embeddings(model, batch, device):
    """
    Utility to compute embeddings.
    Args:
        model: image classification model
        batch: batch of images, stylizations, labels
    Returns:
        embeddings
    """
    inputs = batch[0]
    out = model(inputs.to(device))
    embeddings, logits = out[-1].detach().cpu(), out[0].detach().cpu()
    # print(embeddings.shape, logits.shape)
    return {"embedding": embeddings, 'logits': logits}

MODEL_TYPE = ['simple_odin', 'stsaclf', 'contrastive'][1]
def make_embeddings(
    df,
    model,
    batch_sampler,
    device
):
    """
    Compute embeddings using huggingface models.
    Args:
        df: dataframe with images
        image_name: name of the image column in the dataset
        modelname: huggingface model name
        batched: whether to compute embeddings in batches
        batch_size: batch size
    Returns:
        new dataframe with embeddings
    """
    dataset = train_dataset
    BSZ = 32
    train_loader = DataLoader(dataset = dataset, shuffle = True, batch_size = BSZ)
    embed_batches, logit_batches = [], []
    for batch in train_loader:
        embed, logit = extract_embeddings(art_model, batch, device).values()
        embed_batches.append(embed)
        logit_batches.append(logit)
    embed_batches = torch.cat(embed_batches, dim=0)
    logit_batches = torch.cat(logit_batches, dim=0)
    df_temp = df.assign(embedding=embed_batches.tolist())
    df_temp = df_temp.assign(logit=logit_batches.tolist())
    
    df_emb = pd.DataFrame()
    df_emb["embedding"] = df_temp["embedding"]
    df_emb['logit'] = df_temp['logit']
    return df_emb

    


df = make_kaokore_df(train_dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if MODEL_TYPE == 'simple_odin':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).to(device)
    art_model = model
    art_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True, device = 'cuda')
elif MODEL_TYPE == 'stsaclf':
    art_model = stclf_model.to(device)
    
    
batch_sampler.sampling_probs = torch.load('saves/sampler_odin_VA_Both_27.pt')

embeddings_df = make_embeddings(df, art_model, batch_sampler, device)

art_model.load_state_dict(torch.load('saves/model_stsaclf_27.pth'))
embeddings_df_ft = make_embeddings(df, art_model, batch_sampler, device) 

df["embedding_ft"] = embeddings_df_ft["embedding"]
df['logit_ft'] = embeddings_df_ft['logit']
batch_sampler.sampling_probs = torch.load('saves/sampler_odin_VA_Both_27.pt')
df['sampling_ft'] = batch_sampler.sampling_probs.tolist()

df["embedding_foundation"] = embeddings_df["embedding"]
df['logit_foundation'] = embeddings_df['logit']
batch_sampler.sampling_probs = torch.load('saves/sampler_odin_VA_Both_27.pt')
df["sampling_foundation"] = batch_sampler.sampling_probs.tolist()








def nearest_neighbor_annoy(
    df, embedding_name="embedding", threshold=0.3, tree_size=100
):
    """
    Find nearest neighbor using annoy.
    Args:
        df: dataframe with embeddings
        embedding_name: name of the embedding column
        threshold: threshold for outlier detection
        tree_size: tree size for annoy
    Returns:
        new dataframe with nearest neighbor information
    """
    embs = df[embedding_name]
    print(len(embs[0]))
    t = AnnoyIndex(len(embs[0]), "angular")
    for idx, x in enumerate(embs):
        t.add_item(idx, x)
    t.build(tree_size)
    images = df["image"]
    df_nn = pd.DataFrame()
    nn_id = [t.get_nns_by_item(i, 2)[1] for i in range(len(embs))]
    df_nn["nn_id"] = nn_id
    df_nn["nn_image"] = [images[i] for i in nn_id]
    df_nn["nn_distance"] = [t.get_distance(i, nn_id[i]) for i in range(len(embs))]
    df_nn["nn_flag"] = df_nn.nn_distance < threshold
    return df_nn

def outlier_score_by_embeddings_cleanlab(df, embedding_name="embedding"):
    """
    Calculate outlier score by embeddings using cleanlab
        Args:
            df: dataframe with embeddings
            embedding_name: name of the column with embeddings
        Returns:
            new df_out: dataframe with outlier score
    """
    embs = np.stack(df[embedding_name].to_numpy())
    ood = OutOfDistribution()
    ood_train_feature_scores = ood.fit_score(features=np.stack(embs))
    df_out = pd.DataFrame()
    df_out["outlier_score_embedding"] = ood_train_feature_scores
    return df_out

df_nn = nearest_neighbor_annoy(
    df, embedding_name="embedding_ft", threshold=0.3, tree_size=100
)
df["nn_image"] = df_nn["nn_image"]



df["outlier_score_ft"] = outlier_score_by_embeddings_cleanlab(
                                    df, embedding_name="embedding_ft"
                                )["outlier_score_embedding"]
df["outlier_score_found"] = outlier_score_by_embeddings_cleanlab(
                                df, embedding_name="embedding_foundation"
                                )["outlier_score_embedding"]



# df["label_str"] = df["labels"].apply(lambda x: ds.features["labels"].int2str(x))
dtypes = {
    "nn_image": spotlight.Image,
    "image": spotlight.Image,
    "embedding_ft": spotlight.Embedding,
    "embedding_foundation": spotlight.Embedding,
}
spotlight.show(
    df,
    dtype=dtypes,
    layout="https://spotlight.renumics.com/resources//layout_pre_post_ft.json",
)

