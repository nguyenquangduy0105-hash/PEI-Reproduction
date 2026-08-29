import ast
import math

import pandas as pd


PAD_ERROR = 2
CORRECT = 0
ERROR = 1


def levenshtein_align(canonical, transcript):
    """
    Align canonical phoneme sequence với transcript phoneme sequence.

    Returns:
        aligned_canonical
        aligned_transcript

    Mỗi phần tử có thể là None nếu xảy ra insertion/deletion.
    """
    n = len(canonical)
    m = len(transcript)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if canonical[i - 1] == transcript[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    aligned_canonical = []
    aligned_transcript = []

    i = n
    j = m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if canonical[i - 1] == transcript[j - 1] else 1

            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned_canonical.append(canonical[i - 1])
                aligned_transcript.append(transcript[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_canonical.append(canonical[i - 1])
            aligned_transcript.append(None)
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned_canonical.append(None)
            aligned_transcript.append(transcript[j - 1])
            j -= 1
            continue

        raise RuntimeError("Alignment backtracking failed")

    aligned_canonical.reverse()
    aligned_transcript.reverse()

    return aligned_canonical, aligned_transcript


def make_error_gt(canonical, transcript):
    """
    Tạo error label theo từng canonical phoneme.

    0 = correct
    1 = error
    """
    aligned_canonical, aligned_transcript = levenshtein_align(
        canonical,
        transcript
    )

    error_gt = []

    for c, t in zip(aligned_canonical, aligned_transcript):
        if c is not None:
            if t == c:
                error_gt.append(CORRECT)
            else:
                error_gt.append(ERROR)
        else:
            pass

    return error_gt


def parse_phonemes(x):
    """
    Parse phoneme data from CSV.

    Supported formats:
    1. Python list string:
       "['t', 'ah', 'k']"

    2. Space-separated phoneme string:
       "t ah k"

    3. Existing list/tuple:
       ['t', 'ah', 'k']
    """
    if x is None:
        return []

    if isinstance(x, float) and math.isnan(x):
        return []

    if isinstance(x, (list, tuple)):
        return list(x)

    if not isinstance(x, str):
        raise TypeError(f"Unsupported phoneme value type: {type(x).__name__}")

    x = x.strip()

    if not x:
        return []

    if x.startswith("[") and x.endswith("]"):
        parsed = ast.literal_eval(x)

        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list-like phoneme data, got: {parsed!r}")

        return [str(item) for item in parsed]

    return x.split()


def preprocess_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_columns = {"Canonical", "Transcript"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_csv}: "
            f"{sorted(missing_columns)}"
        )

    error_labels = []

    for idx, row in df.iterrows():
        canonical = parse_phonemes(row["Canonical"])
        transcript = parse_phonemes(row["Transcript"])

        error_gt = make_error_gt(
            canonical,
            transcript
        )

        if len(error_gt) != len(canonical):
            raise RuntimeError(
                f"Alignment error at row {idx}: "
                f"canonical={len(canonical)}, "
                f"error_gt={len(error_gt)}, "
                f"canonical_seq={canonical}, "
                f"transcript_seq={transcript}"
            )

        error_labels.append(error_gt)

    df["Error_GT"] = [
        str(x)
        for x in error_labels
    ]

    df.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved: {output_csv}")
    print(f"Samples: {len(df)}")


if __name__ == "__main__":
    preprocess_csv(
        input_csv="l2_arctic_train_id_old.csv",
        output_csv="train_with_error_old.csv"
    )

    preprocess_csv(
        input_csv="l2_arctic_dev_id_old.csv",
        output_csv="dev_with_error_old.csv"
    )

    preprocess_csv(
        input_csv="l2_arctic_test_id_old.csv",
        output_csv="test_with_error_old.csv"
    )

    preprocess_csv(
        input_csv="l2_arctic_train_id_new.csv",
        output_csv="train_with_error_new.csv"
    )

    preprocess_csv(
        input_csv="l2_arctic_dev_id_new.csv",
        output_csv="dev_with_error_new.csv"
    )

    preprocess_csv(
        input_csv="l2_arctic_test_id_new.csv",
        output_csv="test_with_error_new.csv"
    )