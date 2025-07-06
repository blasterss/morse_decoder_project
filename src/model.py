import torch
from torch import nn
import torch.nn.functional as F

class MorseNet(nn.Module):
    def __init__(self, conv_input_channels: int, conv_output_channels: int,
                 rnn_hidden: int, rnn_layers: int, rnn_type: str,
                 rnn_dropout: float, linear_dropout: float, num_classes: int):
        super().__init__()

        # Свертка для получения информации для каждого временного шага
        self.conv1 = nn.Sequential(
            nn.Conv1d(conv_input_channels, conv_output_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm1d(conv_output_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # Каскадные свертки для выявления различных зависимостей в данных
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_output_channels // 2, conv_output_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(conv_output_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(conv_output_channels, conv_output_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(conv_output_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(conv_output_channels, conv_output_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv_output_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_output_channels, conv_output_channels*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv_output_channels*2),
            nn.ReLU(inplace=True),

            nn.Conv1d(conv_output_channels*2, conv_output_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_output_channels*2),
            nn.ReLU(inplace=True),

            nn.Conv1d(conv_output_channels*2, conv_output_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_output_channels*2),
            nn.ReLU(inplace=True),
        )

        rnn_class = nn.GRU if rnn_type == 'GRU' else nn.LSTM

        self.birnn1 = rnn_class(
            input_size=conv_output_channels*2,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout
        )
        
        self.birnn2 = rnn_class(
            input_size=rnn_hidden * 2,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers // 2,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout
        )

        self.norm = nn.LayerNorm(rnn_hidden * 2) # Нормализация

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(linear_dropout),
            nn.Linear(128, num_classes + 1)  # +1 for CTC blank
        )

    def forward(self, x: torch.Tensor
                ) -> torch.Tensor:         # x: (B, T, 1)
        x = x.permute(0, 2, 1)    # (B, 1, T)
        
        x = self.conv1(x)         # (B, C, T)
        x = self.conv2(x)         # (B, C, T)
        x = self.conv3(x)
        
        x = x.permute(0, 2, 1)    # (B, T, C)
        
        x, _ = self.birnn1(x)     # (B, T, 2*H)
        x, _ = self.birnn2(x)
        x = self.norm(x)
        
        x = self.classifier(x)    # (B, T, num_classes + 1)
        return F.log_softmax(x, dim=-1)