from .utils import morse_labels_to_text, get_prediction
from .model import MorseNet

import torch
from torch.nn import CTCLoss

from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance


def train_model(model: MorseNet, 
                criterion: CTCLoss, 
                optimizer: torch.optim.Adam, 
                scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
                dataloaders,
                device, 
                num_epochs: int = 10):
    '''
    Функция тренировки модели
    '''
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()

            total_loss = 0.0
            total_levenshtein = 0
            total_samples = 0

            for inputs, labels, inp_len, lab_len in tqdm(dataloaders[phase], desc=f"{phase.title()}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                inp_len_tensor = torch.tensor(inp_len, dtype=torch.int32).to(device)
                lab_len_tensor = torch.tensor(lab_len, dtype=torch.int32).to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)                 # (B, T, C)
                    outputs_ctc = outputs.permute(1, 0, 2)  # (T, B, C)

                    loss = criterion(outputs_ctc, labels, inp_len_tensor, lab_len_tensor)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # стабилизация
                        optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Метрика: Levenshtein
                with torch.no_grad():
                    outputs_decoded = outputs_ctc.permute(1, 0, 2)  # (B, T, C)
                    pred_indices_batch = torch.argmax(outputs_decoded, dim=-1).cpu().tolist()

                    labels_cpu = labels.cpu().tolist()
                    start = 0
                    for i in range(len(lab_len)):
                        target_indices = labels_cpu[start:start + lab_len[i]]
                        start += lab_len[i]

                        true_text = morse_labels_to_text(target_indices)

                        pred_indices = pred_indices_batch[i]

                        predicted_text = get_prediction(pred_indices)

                        dist = levenshtein_distance(predicted_text, true_text)
                        total_levenshtein += dist

            avg_loss = total_loss / total_samples
            avg_levenshtein = total_levenshtein / total_samples

            print(f"{phase.title()} | Loss: {avg_loss:.4f} | Avg Levenshtein: {avg_levenshtein:.4f}")

            if phase == 'validation':
                scheduler.step(avg_levenshtein)
