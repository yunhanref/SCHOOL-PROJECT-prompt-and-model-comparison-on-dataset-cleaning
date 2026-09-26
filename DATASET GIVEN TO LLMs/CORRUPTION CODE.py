#BECAUSE THE ORIGINAL DATASET IS CLEAR ENOUGH, I MADE IT MORE DIRTY SO LLMS PERFORMANCE COULD VARY


import pandas as pd
import numpy as np

# ============================================================
# 1. ORIGINAL DATA
# ============================================================

df = pd.read_csv("data.csv")

# Orijinal veriyi koru
clean_df = df.copy()

# Tekrarlanabilirlik
np.random.seed(42)

print("Original shape:", df.shape)


# ============================================================
# 2. OUTLIERS
# ============================================================
# Sayısal feature'ların küçük bir kısmına ekstrem değerler ekle.
# Verinin yaklaşık %1'i etkilenir.

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# id'yi outlier işleminden çıkar
numeric_feature_cols = [
    col for col in numeric_cols
    if col not in ["id"]
]

outlier_rate = 0.01

n_outliers = int(
    len(df) * len(numeric_feature_cols) * outlier_rate
)

for _ in range(n_outliers):

    row = np.random.randint(0, len(df))
    col = np.random.choice(numeric_feature_cols)

    # Mevcut değerin yerine ekstrem ama sayısal bir değer koy
    original_value = df.loc[row, col]

    # Feature'ın standart sapmasını kullan
    std = df[col].std()

    # +/- 5-10 standart sapmalık değer
    direction = np.random.choice([-1, 1])
    multiplier = np.random.uniform(5, 10)

    df.loc[row, col] = (
        original_value + direction * multiplier * std
    )


# ============================================================
# 3. MISSING VALUES / NONE
# ============================================================
# Verinin yaklaşık %2'sine NaN/None ekle.
# Çok fazla veri kaybetmemek için oran düşük tutuluyor.

missing_rate = 0.02

n_missing = int(
    len(df) * len(numeric_feature_cols) * missing_rate
)

for _ in range(n_missing):

    row = np.random.randint(0, len(df))
    col = np.random.choice(numeric_feature_cols)

    # Hem NaN hem None kullan
    if np.random.rand() < 0.5:
        df.loc[row, col] = np.nan
    else:
        df.loc[row, col] = None


# ============================================================
# 4. STRING NOISE
# ============================================================
# Bazı numeric sütunlara string değerler ekle.
# Bu durum LLM'in data type problemini fark etmesini gerektirir.

string_noise_rate = 0.005

n_string_noise = int(
    len(df) * len(numeric_feature_cols) * string_noise_rate
)

noise_values = [
    "unknown",
    "N/A",
    "missing",
    "?",
    "error"
]

for _ in range(n_string_noise):

    row = np.random.randint(0, len(df))
    col = np.random.choice(numeric_feature_cols)

    df.loc[row, col] = np.random.choice(noise_values)


# ============================================================
# 5. DUPLICATE ROWS
# ============================================================
# Dataset'in çok küçük bir bölümünü duplicate yap.
# Toplam gözlem sayısını ciddi şekilde değiştirmiyoruz.

duplicate_rate = 0.01

n_duplicates = int(len(df) * duplicate_rate)

duplicate_rows = df.sample(
    n=n_duplicates,
    random_state=42
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)


# ============================================================
# 6. IRRELEVANT / USELESS COLUMNS
# ============================================================
# LLM'in gereksiz kolonları tespit etmesini test etmek için
# anlamsız kolonlar ekliyoruz.

df["Unnamed: 32"] = np.nan

df["random_noise"] = np.random.randint(
    0,
    100000,
    size=len(df)
)

df["irrelevant_feature"] = np.random.normal(
    loc=0,
    scale=1,
    size=len(df)
)

df["constant_column"] = 1


# ============================================================
# 7. COLUMN NAME NOISE
# ============================================================
# Bazı kolon isimlerine gereksiz boşluk ekle.

columns_to_modify = np.random.choice(
    numeric_feature_cols,
    size=3,
    replace=False
)

for col in columns_to_modify:

    df.rename(
        columns={col: "  " + col + "  "},
        inplace=True
    )


# ============================================================
# 8. SHUFFLE
# ============================================================
# Satır sırasını karıştır.

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 9. SAVE CORRUPTED DATASET
# ============================================================

df.to_csv(
    "breast_cancer_corrupted.csv",
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n========== CORRUPTION SUMMARY ==========")

print("Original shape :", clean_df.shape)
print("Corrupted shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum().sum())

print("\nDuplicated rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes.value_counts())

print("\nColumns:")
print(df.columns.tolist())

print("\nSaved as:")
print("breast_cancer_corrupted.csv")
