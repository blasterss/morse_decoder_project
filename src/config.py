import torch
from pathlib import Path

# --- Внешние файлы ---
PROJECT_ROOT = Path(__file__).parent.parent

AUDIO_DIR = PROJECT_ROOT / "morse_dataset" / "morse_dataset"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

SUBMISSION_NAME = "submission_new_v3.csv"
SUBMISSION_PATH = SUBMISSION_DIR / SUBMISSION_NAME

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test.csv"
SUB_DATA_PATH = PROJECT_ROOT / "data" / "sample_submission.csv"

PREP_DATA_NAME = "preprocess_data_new_v2.csv"
PREP_DATA_PATH = PROJECT_ROOT / "prep_data" / PREP_DATA_NAME

# --- Параметры предобработки ---
SAMPLE_RATE = 8000
N_MELS = 60
N_FFT = 512
HOP_LENGTH = 128
POWER = 13
TARGET_DURATION_SEC = 8

# --- Параметры модели ---
CNN_OUT_CHANNELS = 64
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 4
RNN_TYPE = 'GRU'
RNN_DROPOUT = 0.2
LINEAR_DROPOUT = 0.25

# --- Параметры обучения ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30 
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10

# --- Общие Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MORSE_REVERSE_DICT = {
    # Кодировка алфавита
    ".-": "А", "-...": "Б", ".--": "В", "--.": "Г", "-..": "Д",
    ".": "Е", "...-": "Ж", "--..": "З", "..": "И", ".---": "Й",
    "-.-": "К", ".-..": "Л", "--": "М", "-.": "Н", "---": "О",
    ".--.": "П", ".-.": "Р", "...": "С", "-": "Т", "..-": "У",
    "..-.": "Ф", "....": "Х", "-.-.": "Ц", "---.": "Ч", "----": "Ш",
    "--.-": "Щ", ".--.-.": "Ъ", "-.--": "Ы", "-..-": "Ь", "...-...": "Э", 
    "..--": "Ю", ".-.-": "Я", 

    # Кодировка цифр
    "-----": "0", ".----": "1", "..---": "2",  "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7", 
    "---..": "8", "----.": "9",

    # Кодировка специальных символов
    "|": " ", "#": "#"
}
MORSE_DICT = {v: k for k, v in MORSE_REVERSE_DICT.items()}

CHAR2IDX = {'.': 0, '-': 1, ' ': 2, '|': 3, '#': 4}
IDX2CHAR = {0: '.', 1: '-', 2: ' ', 3: '|', 4: '#'}

NUM_CLASSES = 5
