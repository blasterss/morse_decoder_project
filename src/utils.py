from .config import (MORSE_DICT, MORSE_REVERSE_DICT, 
                     CHAR2IDX, IDX2CHAR, AUDIO_DIR,
                     DEVICE)
from .model import MorseNet
from pandas import DataFrame
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance

import os
import json

def text_to_morse_labels(text: str
                         ) -> list:
    seq = []
    for _, char in enumerate(text.upper()):
        morse = MORSE_DICT.get(char, '')
        for symbol in morse:
            seq.append(CHAR2IDX[symbol])
        seq.append(CHAR2IDX[' '])
    if seq:
        while seq and seq[-1] == CHAR2IDX[' ']:
            seq.pop()
    return seq

def morse_labels_to_text(labels: list
                         ) -> str:
    text = ''.join([IDX2CHAR[i] for i in labels])
    decoded = ""
    words = text.strip().split("|")
    for word in words:
        letters = word.strip().split(" ")
        for letter in letters:
            decoded += MORSE_REVERSE_DICT.get(letter, "#")
        decoded += " "
    return decoded.strip()

def get_prediction(pred_indices: list[int]
                   ) -> str:
    '''
    Функция получения текста из предсказаний модели
    '''
    pred_filtered = []
    last = None
    for p in pred_indices:
        if p != last and p != len(IDX2CHAR):  # убираем blank и повторы
            pred_filtered.append(p)
        last = p

    predicted_text = morse_labels_to_text(pred_filtered)

    return predicted_text

def preprocess_data(df: DataFrame,
                    dst_path: str,
                    extract_func: Callable[[str], tuple[np.ndarray, int]]):

    morse_feats_col = []
    morse_text_col = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['id']
        text = row['message']
        path = os.path.join(AUDIO_DIR, filename)

        try:
            morse_feats, _ = extract_func(path, True)
            morse_feats_col.append(json.dumps(morse_feats.tolist()))
            morse_text = text_to_morse_labels(text)
            morse_text_col.append(json.dumps(morse_text))
        except Exception as e:
            print(f"Ошибка в {filename}: {e}")
            morse_feats_col.append(json.dumps([]))
            morse_text_col.append(json.dumps([]))

    df['morse_feats'] = morse_feats_col
    df['morse_text'] = morse_text_col

    df.to_csv(dst_path, index=False)

def process_dataframe(df: DataFrame,
                      model: MorseNet,
                      extract_func: Callable[[str], tuple[np.ndarray, int]]):
    '''
    Обработка датафрейма с получением предсказанных значений и подсчета расстояние Левенштайна
    '''
    df = df.copy()
    preds = []
    dists = []

    model.eval() 

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['id']
        true_text = row['message']
        path = os.path.join(AUDIO_DIR, filename)

        try:
            morse_features, _ = extract_func(path, True)  # (T, 5)
            input_tensor = torch.tensor(morse_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 5)

            with torch.no_grad():
                output = model(input_tensor)  # (B, T, C)
                pred_indices = torch.argmax(output, dim=-1).squeeze(0).cpu().tolist()  # (T)

            decoded = get_prediction(pred_indices)
            
        except Exception as e:
            decoded = "[ERROR]"
            print(f"[!] Ошибка с файлом {filename}: {e}")

        dist = levenshtein_distance(decoded, true_text)

        preds.append(decoded)
        dists.append(dist)

    df['predicted'] = preds
    df['levenshtein'] = dists

    return df

def get_submission(sub_df: DataFrame,
                   model: MorseNet,
                   extract_func: Callable[[str], tuple[np.ndarray, int]]):
    model.eval()
    total_len = len(sub_df)
    limited_time = True
    for idx, row in tqdm(sub_df.iterrows(), total=total_len):
        filename = row['id']
        path = os.path.join(AUDIO_DIR, filename)
        if idx >= (total_len - 17):
            limited_time = False
        try:
            morse_features, _ = extract_func(path, limited_time)  # (T, 5)
            input_tensor = torch.tensor(morse_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 5)

            with torch.no_grad():
                output = model(input_tensor)  # (B, T, C)
                pred_indices = torch.argmax(output, dim=-1).squeeze(0).cpu().tolist()  # (T)

            decoded = get_prediction(pred_indices)

        except Exception as e:
            decoded = "[ERROR]"
            print(f"[!] Ошибка с файлом {filename}: {e}")

        sub_df.at[idx, 'message'] = decoded

    return sub_df