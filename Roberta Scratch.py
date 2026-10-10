# ============================================================
# ROBERTA BINARY CLASSIFIER TRAINED FROM RANDOM WEIGHTS
#
# Model weights:
#   Randomly initialized. No pretrained RoBERTa model weights.
#
# Tokenizer:
#   Uses the roberta-base tokenizer vocabulary.
#
# Label mapping:
#   true, mostly-true, half-true -> 0
#   false, barely-true, pants-fire -> 1
# ============================================================

import json
import random
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_FILE = Path(
    r"C:\Users\doruk\OneDrive\Desktop\train.tsv"
)

VALID_FILE = Path(
    r"C:\Users\doruk\OneDrive\Desktop\valid.tsv"
)

TEST_FILE = Path(
    r"C:\Users\doruk\OneDrive\Desktop\test.tsv"
)

OUTPUT_DIRECTORY = Path(
    r"C:\Users\doruk\OneDrive\Desktop\roberta_scratch_output"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

TOKENIZER_NAME = "roberta-base"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

NUMBER_OF_EPOCHS = 5

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42

# Reduced RoBERTa architecture.
NUMBER_OF_LAYERS = 4
HIDDEN_DIMENSION = 384
NUMBER_OF_ATTENTION_HEADS = 6
INTERMEDIATE_DIMENSION = 1536

HIDDEN_DROPOUT_RATE = 0.20
ATTENTION_DROPOUT_RATE = 0.20

GRADIENT_CLIP_VALUE = 1.0

# Keep this at 0 initially on Windows.
NUM_WORKERS = 0


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(RANDOM_SEED)


# ============================================================
# 4. REQUIRE CUDA GPU
# ============================================================

print("=" * 70)
print("PYTHON, PYTORCH AND GPU INFORMATION")
print("=" * 70)

print("Python interpreter:", sys.executable)
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "\nCUDA is not available.\n\n"
        "This script is configured to require an NVIDIA GPU.\n"
        "Check that:\n"
        "1. CUDA-enabled PyTorch is installed.\n"
        "2. VS Code is using the correct Python interpreter.\n"
        "3. Your NVIDIA driver is installed and current.\n"
        "4. nvidia-smi works in the VS Code terminal.\n"
    )

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(0)

print("Selected device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(
    "Total GPU memory:",
    round(gpu_memory, 2),
    "GB",
)

# These settings can improve performance on supported GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# PyTorch documents torch.cuda.is_available() as the standard
# availability check for CUDA.
# The model and every training tensor are moved to DEVICE below.


# ============================================================
# 5. CHECK DATASET FILES
# ============================================================

for file_path in [
    TRAIN_FILE,
    VALID_FILE,
    TEST_FILE,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found:\n{file_path}"
        )

print("\nAll dataset files were found.")


# ============================================================
# 6. LIAR DATASET COLUMN NAMES
# ============================================================

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


# ============================================================
# 7. LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    # Class 0: more reliable
    "true": 0,
    "mostly-true": 0,
    "mostly true": 0,
    "half-true": 0,
    "half true": 0,

    # Class 1: less reliable
    "false": 1,
    "barely-true": 1,
    "barely true": 1,
    "pants-fire": 1,
    "pants fire": 1,
    "pants-on-fire": 1,
    "pants on fire": 1,

    # Already binary
    "0": 0,
    "1": 1,
    "0.0": 0,
    "1.0": 1,
}

KNOWN_LABELS = set(LABEL_MAPPING.keys())


# ============================================================
# 8. DATA-READING FUNCTIONS
# ============================================================

def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def file_appears_to_have_header(
    file_path: Path,
) -> bool:
    first_row = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        nrows=1,
        dtype=str,
        keep_default_na=False,
    )

    first_values = {
        normalize_column_name(value)
        for value in first_row.iloc[0].tolist()
    }

    possible_headers = {
        "id",
        "label",
        "labels",
        "class",
        "target",
        "statement",
        "text",
        "content",
        "claim",
        "sentence",
        "subject",
        "speaker",
    }

    return bool(
        first_values.intersection(possible_headers)
    )


def read_tsv_file(
    file_path: Path,
) -> pd.DataFrame:
    has_header = file_appears_to_have_header(
        file_path
    )

    if has_header:
        dataframe = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
        )

        dataframe.columns = [
            normalize_column_name(column)
            for column in dataframe.columns
        ]

    else:
        dataframe = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
        )

        number_of_columns = dataframe.shape[1]

        if number_of_columns == len(LIAR_COLUMNS):
            dataframe.columns = LIAR_COLUMNS

        elif number_of_columns >= 3:
            generated_columns = [
                f"column_{index}"
                for index in range(number_of_columns)
            ]

            dataframe.columns = generated_columns

            dataframe = dataframe.rename(
                columns={
                    generated_columns[1]: "label",
                    generated_columns[2]: "statement",
                }
            )

        elif number_of_columns == 2:
            first_values = set(
                dataframe.iloc[:, 0]
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )

            second_values = set(
                dataframe.iloc[:, 1]
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )

            first_is_label = first_values.issubset(
                KNOWN_LABELS
            )

            second_is_label = second_values.issubset(
                KNOWN_LABELS
            )

            if first_is_label and not second_is_label:
                dataframe.columns = [
                    "label",
                    "statement",
                ]

            elif second_is_label and not first_is_label:
                dataframe.columns = [
                    "statement",
                    "label",
                ]

            else:
                dataframe.columns = [
                    "statement",
                    "label",
                ]

        else:
            raise ValueError(
                f"{file_path} does not contain enough columns."
            )

    print(
        f"\nLoaded {file_path.name}: "
        f"{len(dataframe):,} rows"
    )

    print(
        "Columns:",
        dataframe.columns.tolist(),
    )

    return dataframe


def find_named_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
):
    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        normalized_name = normalize_column_name(
            possible_name
        )

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def detect_text_and_label_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[str, str]:
    text_column = find_named_column(
        dataframe,
        [
            "statement",
            "text",
            "claim",
            "content",
            "sentence",
            "news",
            "article",
            "headline",
        ],
    )

    label_column = find_named_column(
        dataframe,
        [
            "label",
            "labels",
            "class",
            "target",
            "rating",
            "verdict",
        ],
    )

    if label_column is None:
        for column in dataframe.columns:
            unique_values = set(
                dataframe[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )

            if (
                unique_values
                and unique_values.issubset(
                    KNOWN_LABELS
                )
            ):
                label_column = column
                break

    if text_column is None:
        candidate_columns = [
            column
            for column in dataframe.columns
            if column != label_column
        ]

        if candidate_columns:
            average_lengths = {
                column: (
                    dataframe[column]
                    .astype(str)
                    .str.len()
                    .mean()
                )
                for column in candidate_columns
            }

            text_column = max(
                average_lengths,
                key=average_lengths.get,
            )

    if text_column is None or label_column is None:
        raise ValueError(
            f"Could not detect text and label columns "
            f"in {dataset_name}.\n"
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    print(
        f"{dataset_name}: "
        f"text='{text_column}', "
        f"label='{label_column}'"
    )

    return text_column, label_column


def convert_label(value) -> int:
    normalized_value = (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
    )

    if normalized_value not in LABEL_MAPPING:
        raise ValueError(
            f"Unknown label: {value!r}"
        )

    return LABEL_MAPPING[normalized_value]


def prepare_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    text_column, label_column = (
        detect_text_and_label_columns(
            dataframe,
            dataset_name,
        )
    )

    prepared = dataframe[
        [text_column, label_column]
    ].copy()

    prepared.columns = [
        "text",
        "original_label",
    ]

    prepared["text"] = (
        prepared["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prepared["original_label"] = (
        prepared["original_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prepared = prepared[
        (prepared["text"] != "")
        & (prepared["original_label"] != "")
    ].copy()

    print(
        f"\nOriginal {dataset_name} labels:",
        sorted(
            prepared["original_label"]
            .unique()
            .tolist()
        ),
    )

    prepared["label"] = (
        prepared["original_label"]
        .apply(convert_label)
        .astype(int)
    )

    prepared = prepared[
        ["text", "label"]
    ].reset_index(drop=True)

    original_size = len(prepared)

    prepared = prepared.drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    duplicate_count = (
        original_size - len(prepared)
    )

    print(
        f"{dataset_name} examples:",
        f"{len(prepared):,}",
    )

    print(
        f"{dataset_name} duplicates removed:",
        duplicate_count,
    )

    print(
        f"{dataset_name} label distribution:"
    )

    print(
        prepared["label"]
        .value_counts()
        .sort_index()
    )

    return prepared


# ============================================================
# 9. LOAD DATA
# ============================================================

train_dataframe = prepare_dataframe(
    read_tsv_file(TRAIN_FILE),
    "training",
)

valid_dataframe = prepare_dataframe(
    read_tsv_file(VALID_FILE),
    "validation",
)

test_dataframe = prepare_dataframe(
    read_tsv_file(TEST_FILE),
    "test",
)


# ============================================================
# 10. DATA-LEAKAGE CHECK
# ============================================================

def normalize_text_for_comparison(
    text: str,
) -> str:
    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


train_texts = set(
    train_dataframe["text"].apply(
        normalize_text_for_comparison
    )
)

valid_texts = set(
    valid_dataframe["text"].apply(
        normalize_text_for_comparison
    )
)

test_texts = set(
    test_dataframe["text"].apply(
        normalize_text_for_comparison
    )
)

train_valid_overlap = train_texts & valid_texts
train_test_overlap = train_texts & test_texts
valid_test_overlap = valid_texts & test_texts

print("\n" + "=" * 70)
print("DATA-LEAKAGE CHECK")
print("=" * 70)

print(
    "Train-validation overlap:",
    len(train_valid_overlap),
)

print(
    "Train-test overlap:",
    len(train_test_overlap),
)

print(
    "Validation-test overlap:",
    len(valid_test_overlap),
)

if (
    train_valid_overlap
    or train_test_overlap
    or valid_test_overlap
):
    print(
        "\nWARNING: Identical text occurs in multiple "
        "dataset splits. This can inflate accuracy."
    )


# ============================================================
# 11. LOAD ROBERTA TOKENIZER
# ============================================================

print("\nLoading RoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
)

print(
    "Tokenizer vocabulary size:",
    len(tokenizer),
)

print(
    "Padding token:",
    tokenizer.pad_token,
)

print(
    "Padding token ID:",
    tokenizer.pad_token_id,
)


# ============================================================
# 12. PYTORCH DATASET
# ============================================================

class FakeNewsDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):
        self.texts = (
            dataframe["text"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": (
                encoding["input_ids"]
                .squeeze(0)
            ),
            "attention_mask": (
                encoding["attention_mask"]
                .squeeze(0)
            ),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


train_dataset = FakeNewsDataset(
    train_dataframe,
    tokenizer,
    MAX_LENGTH,
)

valid_dataset = FakeNewsDataset(
    valid_dataframe,
    tokenizer,
    MAX_LENGTH,
)

test_dataset = FakeNewsDataset(
    test_dataframe,
    tokenizer,
    MAX_LENGTH,
)


# ============================================================
# 13. DATA LOADERS
# ============================================================

train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ============================================================
# 14. CREATE RANDOMLY INITIALIZED ROBERTA
# ============================================================

# IMPORTANT:
#
# This creates RoBERTa from a configuration and therefore
# initializes all neural-network weights randomly.
#
# It does NOT use:
#
# RobertaForSequenceClassification.from_pretrained(
#     "roberta-base"
# )

model_config = RobertaConfig(
    vocab_size=len(tokenizer),

    max_position_embeddings=514,

    type_vocab_size=1,

    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,

    num_hidden_layers=NUMBER_OF_LAYERS,

    hidden_size=HIDDEN_DIMENSION,

    num_attention_heads=(
        NUMBER_OF_ATTENTION_HEADS
    ),

    intermediate_size=(
        INTERMEDIATE_DIMENSION
    ),

    hidden_act="gelu",

    hidden_dropout_prob=(
        HIDDEN_DROPOUT_RATE
    ),

    attention_probs_dropout_prob=(
        ATTENTION_DROPOUT_RATE
    ),

    classifier_dropout=(
        HIDDEN_DROPOUT_RATE
    ),

    initializer_range=0.02,

    layer_norm_eps=1e-5,

    num_labels=2,

    id2label={
        0: "RELIABLE",
        1: "UNRELIABLE",
    },

    label2id={
        "RELIABLE": 0,
        "UNRELIABLE": 1,
    },
)

model = RobertaForSequenceClassification(
    model_config
)

model.to(DEVICE)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    "Model initialized with random weights."
)

print(
    "Transformer layers:",
    NUMBER_OF_LAYERS,
)

print(
    "Hidden dimension:",
    HIDDEN_DIMENSION,
)

print(
    "Attention heads:",
    NUMBER_OF_ATTENTION_HEADS,
)

print(
    "Intermediate dimension:",
    INTERMEDIATE_DIMENSION,
)

print(
    "Total parameters:",
    f"{total_parameters:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}",
)

print(
    "Model device:",
    next(model.parameters()).device,
)


# ============================================================
# 15. OPTIMIZER
# ============================================================

no_decay_terms = [
    "bias",
    "LayerNorm.weight",
    "layer_norm.weight",
]

optimizer_parameter_groups = [
    {
        "params": [
            parameter
            for name, parameter
            in model.named_parameters()
            if not any(
                term in name
                for term in no_decay_terms
            )
        ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [
            parameter
            for name, parameter
            in model.named_parameters()
            if any(
                term in name
                for term in no_decay_terms
            )
        ],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(
    optimizer_parameter_groups,
    lr=LEARNING_RATE,
)


# ============================================================
# 16. LEARNING-RATE SCHEDULER
# ============================================================

total_training_steps = (
    len(train_loader)
    * NUMBER_OF_EPOCHS
)

warmup_steps = int(
    total_training_steps
    * WARMUP_RATIO
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

print(
    "\nTotal possible training steps:",
    total_training_steps,
)

print(
    "Warmup steps:",
    warmup_steps,
)


# ============================================================
# 17. MIXED PRECISION
# ============================================================

use_mixed_precision = True

try:
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_mixed_precision,
    )

    def autocast_context():
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_mixed_precision,
        )

except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_mixed_precision
    )

    def autocast_context():
        return torch.cuda.amp.autocast(
            enabled=use_mixed_precision
        )


# ============================================================
# 18. METRICS
# ============================================================

def calculate_metrics(
    true_labels: list[int],
    predictions: list[int],
) -> dict:
    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(macro_f1),
    }


# ============================================================
# 19. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    scaler,
    device,
) -> dict:
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        attention_mask = (
            batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_VALUE,
        )

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .tolist()
        )

        if (
            batch_number % 50 == 0
            or batch_number == len(data_loader)
        ):
            allocated_memory = (
                torch.cuda.memory_allocated(0)
                / 1024**3
            )

            reserved_memory = (
                torch.cuda.memory_reserved(0)
                / 1024**3
            )

            print(
                f"  Batch {batch_number:>4}/"
                f"{len(data_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"GPU allocated: "
                f"{allocated_memory:.2f} GB | "
                f"GPU reserved: "
                f"{reserved_memory:.2f} GB"
            )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = (
        total_loss / len(data_loader)
    )

    return metrics


# ============================================================
# 20. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
) -> tuple[
    dict,
    list[int],
    list[int],
    np.ndarray,
]:
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        attention_mask = (
            batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with autocast_context():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += outputs.loss.item()

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1,
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = (
        total_loss / len(data_loader)
    )

    probability_array = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return (
        metrics,
        all_labels,
        all_predictions,
        probability_array,
    )


# ============================================================
# 21. TRAINING LOOP
# ============================================================

best_validation_macro_f1 = -1.0
epochs_without_improvement = 0

best_model_path = (
    OUTPUT_DIRECTORY / "best_model.pt"
)

history = []

print("\n" + "=" * 70)
print("STARTING ROBERTA TRAINING")
print("=" * 70)

for epoch in range(
    1,
    NUMBER_OF_EPOCHS + 1,
):
    print(
        f"\nEpoch {epoch}/{NUMBER_OF_EPOCHS}"
    )

    print("-" * 70)

    train_metrics = train_one_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=DEVICE,
    )

    (
        validation_metrics,
        _,
        _,
        _,
    ) = evaluate_model(
        model=model,
        data_loader=valid_loader,
        device=DEVICE,
    )

    epoch_record = {
        "epoch": epoch,

        "train_loss": (
            train_metrics["loss"]
        ),

        "train_accuracy": (
            train_metrics["accuracy"]
        ),

        "train_macro_f1": (
            train_metrics["macro_f1"]
        ),

        "validation_loss": (
            validation_metrics["loss"]
        ),

        "validation_accuracy": (
            validation_metrics["accuracy"]
        ),

        "validation_macro_f1": (
            validation_metrics["macro_f1"]
        ),
    }

    history.append(epoch_record)

    print("\nTraining metrics")

    print(
        f"  Loss:      "
        f"{train_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy:  "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro-F1:  "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print("\nValidation metrics")

    print(
        f"  Loss:      "
        f"{validation_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy:  "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"  Macro-F1:  "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    current_macro_f1 = (
        validation_metrics["macro_f1"]
    )

    if (
        current_macro_f1
        > best_validation_macro_f1
    ):
        best_validation_macro_f1 = (
            current_macro_f1
        )

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
            ),

            "validation_metrics": (
                validation_metrics
            ),

            "model_config": (
                model_config.to_dict()
            ),
        }

        torch.save(
            checkpoint,
            best_model_path,
        )

        print(
            "\nBest model saved:",
            best_model_path,
        )

    else:
        epochs_without_improvement += 1

        print(
            "\nNo validation improvement for",
            epochs_without_improvement,
            "epoch(s).",
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping activated."
            )

            break


# ============================================================
# 22. SAVE TRAINING HISTORY
# ============================================================

history_dataframe = pd.DataFrame(history)

history_file = (
    OUTPUT_DIRECTORY
    / "training_history.csv"
)

history_dataframe.to_csv(
    history_file,
    index=False,
)

print(
    "\nTraining history saved to:",
    history_file,
)


# ============================================================
# 23. PLOT TRAINING HISTORY
# ============================================================

if not history_dataframe.empty:
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["train_loss"],
        marker="o",
        label="Training loss",
    )

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe[
            "validation_loss"
        ],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RoBERTa Training History")
    plt.legend()
    plt.tight_layout()

    history_plot_file = (
        OUTPUT_DIRECTORY
        / "training_history.png"
    )

    plt.savefig(
        history_plot_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# 24. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

try:
    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False,
    )

except TypeError:
    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

print(
    "Best epoch:",
    checkpoint["epoch"],
)

print(
    "Best validation macro-F1:",
    round(
        checkpoint[
            "validation_metrics"
        ]["macro_f1"],
        4,
    ),
)

print(
    "Loaded model device:",
    next(model.parameters()).device,
)


# ============================================================
# 25. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

(
    test_metrics,
    test_true_labels,
    test_predictions,
    test_probabilities,
) = evaluate_model(
    model=model,
    data_loader=test_loader,
    device=DEVICE,
)

for metric_name, metric_value in (
    test_metrics.items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_true_labels,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "Reliable (0)",
            "Unreliable (1)",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    test_true_labels,
    test_predictions,
    labels=[0, 1],
)

print("Confusion matrix:")
print(matrix)

confusion_display = (
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Reliable (0)",
            "Unreliable (1)",
        ],
    )
)

figure, axis = plt.subplots(
    figsize=(7, 6)
)

confusion_display.plot(
    ax=axis,
    values_format="d",
)

plt.title(
    "RoBERTa From-Scratch Test Confusion Matrix"
)

plt.tight_layout()

confusion_matrix_file = (
    OUTPUT_DIRECTORY
    / "test_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 28. SAVE TEST PREDICTIONS
# ============================================================

prediction_dataframe = (
    test_dataframe.copy()
)

prediction_dataframe[
    "predicted_label"
] = test_predictions

prediction_dataframe[
    "true_label_name"
] = prediction_dataframe["label"].map(
    {
        0: "RELIABLE",
        1: "UNRELIABLE",
    }
)

prediction_dataframe[
    "predicted_label_name"
] = prediction_dataframe[
    "predicted_label"
].map(
    {
        0: "RELIABLE",
        1: "UNRELIABLE",
    }
)

prediction_dataframe[
    "probability_reliable"
] = test_probabilities[:, 0]

prediction_dataframe[
    "probability_unreliable"
] = test_probabilities[:, 1]

prediction_dataframe[
    "correct"
] = (
    prediction_dataframe["label"]
    == prediction_dataframe[
        "predicted_label"
    ]
)

prediction_file = (
    OUTPUT_DIRECTORY
    / "test_predictions.csv"
)

prediction_dataframe.to_csv(
    prediction_file,
    index=False,
)

print(
    "\nTest predictions saved to:",
    prediction_file,
)


# ============================================================
# 29. DISPLAY INCORRECT PREDICTIONS
# ============================================================

incorrect_predictions = (
    prediction_dataframe[
        prediction_dataframe["correct"] == False
    ]
    .copy()
)

incorrect_predictions[
    "confidence"
] = incorrect_predictions[
    [
        "probability_reliable",
        "probability_unreliable",
    ]
].max(axis=1)

incorrect_predictions = (
    incorrect_predictions.sort_values(
        by="confidence",
        ascending=False,
    )
)

print("\n" + "=" * 70)
print("FIRST 20 INCORRECT PREDICTIONS")
print("=" * 70)

if incorrect_predictions.empty:
    print(
        "No incorrect predictions were found."
    )

else:
    for error_number, (_, row) in enumerate(
        incorrect_predictions.head(20).iterrows(),
        start=1,
    ):
        print(
            f"\nError {error_number}"
        )

        print("-" * 60)
        print("Text:", row["text"])

        print(
            "True label:",
            row["true_label_name"],
        )

        print(
            "Predicted label:",
            row["predicted_label_name"],
        )

        print(
            "Reliable probability:",
            round(
                row["probability_reliable"],
                4,
            ),
        )

        print(
            "Unreliable probability:",
            round(
                row[
                    "probability_unreliable"
                ],
                4,
            ),
        )

        print(
            "Confidence:",
            round(
                row["confidence"],
                4,
            ),
        )


# ============================================================
# 30. SAVE MODEL IN HUGGING FACE FORMAT
# ============================================================

final_model_directory = (
    OUTPUT_DIRECTORY / "final_model"
)

final_model_directory.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    final_model_directory
)

tokenizer.save_pretrained(
    final_model_directory
)

metrics_file = (
    OUTPUT_DIRECTORY / "test_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_metrics,
        file,
        indent=4,
    )


# ============================================================
# 31. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(
    "Best checkpoint:",
    best_model_path,
)

print(
    "Final model directory:",
    final_model_directory,
)

print(
    "Test predictions:",
    prediction_file,
)

print(
    "Test metrics:",
    metrics_file,
)

print(
    "Confusion matrix image:",
    confusion_matrix_file,
)

print(
    "Training history:",
    history_file,
)

print(
    "\nFinal CUDA memory allocated:",
    round(
        torch.cuda.memory_allocated(0)
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "Final CUDA memory reserved:",
    round(
        torch.cuda.memory_reserved(0)
        / 1024**3,
        2,
    ),
    "GB",
)