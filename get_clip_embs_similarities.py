import os
import numpy as np


def compute_similarity(img_path, txt_path):
    img = np.load(img_path)
    txt = np.load(txt_path)
    return np.einsum('ij,ij->i', img, txt)

img_emb = '/fsx/marianna/translate_dataset/embeddings_folder/img_emb/img_emb_0.npy'
txt_emb = '/fsx/marianna/translate_dataset/embeddings_folder/text_emb/text_emb_0.npy'

img_emb_en = '/fsx/marianna/translate_dataset/embeddings_folder_en/img_emb/img_emb_0.npy'
txt_emb_en = '/fsx/marianna/translate_dataset/embeddings_folder_en/text_emb/text_emb_0.npy'

sim = compute_similarity(img_emb, txt_emb)
sim_en = compute_similarity(img_emb_en, txt_emb_en)
sorted_sim = np.sort(sim)
sorted_sim_en = np.sort(sim_en)


print('Similarity for original dataset:')
for i in range(1,10, 1):
    q = sorted_sim[i* int(len(sim)/10)]
    print(f'{i*10}% quantile -  {q}')

print('Similarity for translated dataset:')
for i in range(1,10, 1):
    q = sorted_sim_en[i* int(len(sim)/10)]
    print(f'{i*10}% quantile  -  {q}')


