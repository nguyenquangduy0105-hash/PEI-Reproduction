"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
import torchaudio
import torchaudio.transforms as t
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
from tqdm import tqdm

from scipy.io import wavfile
from scipy.signal import resample_poly

from Model import pei
import pandas as pd
from Wav2vec2 import wav2vec2

from cfg import cfg
import pickle
import ast
import math

with open ("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

p2idx = vocab["p2idx"]
idx2p = vocab["idx2p"]

def load_audio_without_torchaudio(path, target_sr=16000):
    sr, waveform = wavfile.read(path + ".wav")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if waveform.dtype.kind in {"i", "u"}:
        max_value = max(abs(float(waveform.min())), abs(float(waveform.max())), 1.0)
        waveform = waveform.astype("float32") / max_value
    else:
        waveform = waveform.astype("float32")

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        waveform = resample_poly(waveform, up, down).astype("float32")
        sr = target_sr

    waveform = torch.from_numpy(waveform).unsqueeze(0)

    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        waveform = torch.nan_to_num(waveform)

    return waveform, sr

def parse_sequence(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value

class pei_dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-100h"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = sample["Path"]
        waveform, sr = load_audio_without_torchaudio(path, target_sr=16000)

        waveform_np = waveform.squeeze(0).numpy()
        inputs = self.processor(waveform_np, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        transcript = torch.tensor(
            parse_sequence(sample["Transcript"]),
            dtype=torch.long
        )

        canonical = torch.tensor(
            parse_sequence(sample["Canonical"]),
            dtype=torch.long
        )

        error = torch.tensor(
            parse_sequence(sample["Error_GT"]),
            dtype=torch.long
        )

        return (
            input_values,
            transcript,
            canonical,
            error
        )

pad_idx = p2idx['[PAD]']
def collate_fn(batch, pad_idx=pad_idx):
    input_values = [item[0] for item in batch]
    transcript = [item[1] for item in batch]
    canonical = [item[2] for item in batch]
    error_gt = [item[3] for item in batch]

    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value = 0.0)
    trans_padded = pad_sequence(transcript, batch_first=True, padding_value = pad_idx)
    canonical_padded = pad_sequence(canonical, batch_first=True, padding_value = pad_idx)
    error_gt_padded = pad_sequence(error_gt, batch_first=True, padding_value = 2)

    input_lengths = torch.LongTensor([s.size(0) for s in input_values])
    label_lengths = torch.LongTensor([l.size(0) for l in transcript])

    return (
        input_values_padded,
        trans_padded,
        canonical_padded,
        input_lengths,
        label_lengths,
        error_gt_padded
    )


def weight_init(model):
    if isinstance(model, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(model, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in model.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # Chỉ áp dụng trực giao lên ma trận ẩn-ẩn ẩn
            elif 'bias' in name:
                # Cổng forget của LSTM (vị trí từ hidden_size đến 2*hidden_size) nên được khởi tạo bằng 1
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    elif isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)


import random

def ctc_greedy_decode(probs, idx2p, blank_token=0):
    # Lấy index có xác suất cao nhất tại mỗi time step
    #best_path = torch.argmax(probs, dim=-1).cpu().numpy()  # [Time]
    best_probs, best_path = torch.max(probs, dim=-1)  # [Time]

    # 1. Loại bỏ các phần tử lặp liên tiếp
    collapsed_path = []
    last_token = None
    for token in best_path:
        if token != last_token:
            collapsed_path.append(token)
            last_token = token

    # 2. Loại bỏ blank token (thường là 0) và chuyển sang Phoneme
    phonemes = [idx2p[token.item()] for token in collapsed_path if token.item() != blank_token]

    return phonemes, collapsed_path, best_path, best_probs


def visualize_random_samples(model, dataset, device, idx2p, w2v2, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    CTC_BLANK_ID = 0

    print(f"\n--- 🔍 VIZUALIZING {num_samples} RANDOM SAMPLES ---")

    with torch.no_grad():
        for idx in indices:
            # FIX 2: dataset __getitem__ giờ trả về 4 biến
            waveform, transcript, canon, error = dataset[idx]

            # Đưa về batch size = 1 để model xử lý
            waveform = waveform.unsqueeze(0).to(device)
            target_tokens = transcript
            canon = canon.unsqueeze(0).to(device)
            input_lengths = torch.LongTensor([s.size(0) for s in waveform])
            input_lengths = input_lengths.to(device)


            # Forward pass: FIX 4: Truyền mel_spec vào model
            ctc_logits, nll_logits = model(waveform, canon, w2v2, input_lengths=input_lengths,pad_idx=pad_idx)
            ctc_probs = F.log_softmax(ctc_logits, dim=-1)#.transpose(0, 1)
            print(f"probs shape : {ctc_probs.shape}')")

            # Decode
            pred_phonemes_raw, pred_ID, best_path, best_probs = ctc_greedy_decode(ctc_probs[0], idx2p, blank_token=CTC_BLANK_ID)

            # Lọc sạch token Space (0)
            clean_pred = [p for p in pred_phonemes_raw]
            clean_target = [
                idx2p[token.item()]
                for token in target_tokens
                if token.item() != CTC_BLANK_ID
            ]

            print(f"Sample index: {idx}")
            print(f" > Target: {' '.join(clean_target)}")
            print(f" > Output: {' '.join(clean_pred)}")
            #print(f" > RAW ID: {pred_ID}")
            #print(f" > ARGMAX PATH : {best_path}")
            #print(f" > BEST PROBS : {best_probs}")
            print("-" * 30)





def train(model, iterator, device, criterion_ctc, criterion_nll, optimizer, w2v2, lmd=0.5):
    model.train()
    train_loss = 0
    print("training")
    train_bar = tqdm(enumerate(iterator), total=len(iterator), desc="training")
    for batch_idx, (waveform, transcript, canonical, input_length, label_length, error_gt) in train_bar:
        waveform = waveform.to(device)
        transcript = transcript.to(device)
        canonical = canonical.to(device)
        input_length = input_length.to(device)
        label_length = label_length.to(device)
        error_gt = error_gt.to(device)

        ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length,pad_idx=pad_idx)
        #ctc_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        T = ctc_logits.shape[1]

        input_length = torch.clamp(
            input_length // 320,
            max=T
        )
        ctc_loss = criterion_ctc(ctc_probs, transcript, input_length, label_length)
        nll_loss = criterion_nll(
            nll_logits.reshape(-1,2),
            error_gt.reshape(-1)
        )
        loss = ctc_loss*lmd + (1-lmd)*nll_loss
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

    return train_loss / max(1, len(iterator))

def validate(model, iterator, device, criterion_ctc, criterion_nll, w2v2, lmd=0.5):
    model.eval()
    dev_loss = 0
    print("validating")
    dev_bar = tqdm(enumerate(iterator), total=len(iterator), desc="Validating")
    with torch.no_grad():
        for batch_idx, (waveform, transcript, canonical, input_length, label_length, error_gt) in dev_bar:
            waveform = waveform.to(device)
            transcript = transcript.to(device)
            canonical = canonical.to(device)
            input_length = input_length.to(device)
            label_length = label_length.to(device)
            error_gt = error_gt.to(device)

            ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length,pad_idx=pad_idx)
            #ctc_probs = F.log_softmax(ctc_logits, dim=-1)
            ctc_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            T = ctc_logits.shape[1]

            input_length = torch.clamp(
                input_length // 320,
                max=T
            )
            ctc_loss = criterion_ctc(ctc_probs, transcript, input_length, label_length)
            nll_loss = criterion_nll(
                nll_logits.reshape(-1, 2),
                error_gt.reshape(-1)
            )
            loss = ctc_loss*lmd + (1-lmd)*nll_loss
            dev_loss += loss.item()
            dev_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

    return dev_loss / max(1, len(iterator))

def save_checkpoint(
        model,
        optimizer,
        scheduler=None,
        loss=None,
        epoch=None,
        checkpoint_dir=cfg.CHECKPOINT_DIR,
        filename="checkpoint.pth"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': loss,
        'epoch': epoch
    }

    save_path = os.path.join(checkpoint_dir, filename)

    torch.save(checkpoint, save_path)
    print("Saved Checkpoint")


def get_clean_sequence(tensor_ids, blank_token=0):

    # Chuyển về list thuần túy
    ids = tensor_ids.tolist() if hasattr(tensor_ids, 'tolist') else list(tensor_ids)

    # 1. Loại bỏ các token lặp liên tiếp
    collapsed = []
    if len(ids) > 0:
        collapsed.append(ids[0])
        for i in range(1, len(ids)):
            if ids[i] != ids[i - 1]:
                collapsed.append(ids[i])

    # 2. Loại bỏ blank token (thường là 0)
    final = [x for x in collapsed if x != blank_token]
    return final


import torch
from typing import List, Tuple, Dict, Union


class MDDEvaluator:
    def __init__(self):
        # Bộ đếm cho bài toán Detection & Diagnosis
        self.metrics_counts = {
            "TA": 0,  # True Acceptance
            "FA": 0,  # False Acceptance
            "TR": 0,  # True Rejection
            "FR": 0,  # False Rejection
            "CD": 0,  # Correct Diagnosis
            "DE": 0  # Diagnosis Error
        }

        # Bộ đếm cho bài toán ASR (Tính PER của Output so với Canonical)
        self.asr_counts = {
            "correct": 0,
            "substitution": 0,
            "deletion": 0,
            "insertion": 0,
            "total_canonical": 0
        }

    def _to_list(self, seq: Union[List[int], torch.Tensor]) -> List[int]:
        if isinstance(seq, torch.Tensor):
            return seq.detach().cpu().tolist()
        return list(seq)

    def calculate_edit_distance(self, hyp: Union[List[int], torch.Tensor], ref: Union[List[int], torch.Tensor]) -> \
    Tuple[int, int, int, int, List[Tuple[str, int]]]:

        hyp_list = self._to_list(hyp)
        ref_list = self._to_list(ref)

        n, m = len(hyp_list), len(ref_list)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Khởi tạo ma trận chi phí
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if hyp_list[i - 1] == ref_list[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,  # Substitution
                        dp[i - 1][j] + 1,  # Insertion (Thừa trong hyp)
                        dp[i][j - 1] + 1  # Deletion (Thiếu trong hyp)
                    )

        # Trở ngược đường đi (Backtracking) để gán trạng thái cụ thể cho từng vị trí ref
        i, j = n, m
        sub_cnt, del_cnt, ins_cnt, cor_cnt = 0, 0, 0, 0
        ref_mapping = [None] * m

        while i > 0 or j > 0:
            # Ưu tiên 0: Trùng khớp (Correct)
            if i > 0 and j > 0 and hyp_list[i - 1] == ref_list[j - 1]:
                cor_cnt += 1
                ref_mapping[j - 1] = ("correct", hyp_list[i - 1])
                i -= 1
                j -= 1

            # Ưu tiên 1: Deletion (Đẩy Deletion lên xử lý trước để tránh dồn Sub về cuối câu)
            elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
                del_cnt += 1
                ref_mapping[j - 1] = ("del", -1)
                j -= 1

            # Ưu tiên 2: Insertion (Thừa trong hyp)
            elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
                ins_cnt += 1
                # LƯU Ý: Hiện tại ref_mapping có size tĩnh [m], không lưu được Insertion.
                # Nếu bạn muốn đo MDD chính xác cả lỗi FA do người dùng đọc thừa âm,
                # bạn sẽ phải cấu trúc lại ref_mapping thành List linh hoạt hơn thay vì mảng tĩnh.
                i -= 1

            # Ưu tiên 3: Substitution (Xếp cuối cùng, chỉ chốt khi không còn đường lui)
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                sub_cnt += 1
                ref_mapping[j - 1] = ("sub", hyp_list[i - 1])
                i -= 1
                j -= 1

        return cor_cnt, sub_cnt, del_cnt, ins_cnt, ref_mapping

    def update_sample(self, canonical: Union[List[int], torch.Tensor],
                      transcript: Union[List[int], torch.Tensor],
                      output: Union[List[int], torch.Tensor]):

        canonical_list = self._to_list(canonical)

        # 1. Align Transcript (Người nói thực tế) với Canonical (Chuẩn)
        _, _, _, _, trans_map = self.calculate_edit_distance(transcript, canonical)

        # 2. Align Output (Mô hình dự đoán) với Canonical (Chuẩn)
        o_cor, o_sub, o_del, o_ins, out_map = self.calculate_edit_distance(output, canonical)

        # Cập nhật thông tin tính toán PER cho ASR Head (Dựa trên Output vs Canonical)
        self.asr_counts["correct"] += o_cor
        self.asr_counts["substitution"] += o_sub
        self.asr_counts["deletion"] += o_del
        self.asr_counts["insertion"] += o_ins
        self.asr_counts["total_canonical"] += len(canonical_list)

        # Duyệt qua từng vị trí của chuỗi Canonical để chấm điểm chẩn đoán lỗi phát âm
        for j in range(len(canonical_list)):
            trans_status, trans_val = trans_map[j]
            out_status, out_val = out_map[j]

            is_trans_correct = (trans_status == "correct")
            is_out_correct = (out_status == "correct")

            # --- PHÂN LOẠI 4 KIỂU DETECTION ---
            if is_trans_correct and is_out_correct:
                self.metrics_counts["TA"] += 1  # Người nói ĐÚNG, Model báo ĐÚNG

            elif not is_trans_correct and is_out_correct:
                self.metrics_counts["FA"] += 1  # Người nói SAI, Model báo ĐÚNG

            elif is_trans_correct and not is_out_correct:
                self.metrics_counts["FR"] += 1  # Người nói ĐÚNG, Model báo SAI

            elif not is_trans_correct and not is_out_correct:
                self.metrics_counts["TR"] += 1  # Người nói SAI, Model báo SAI

                # --- PHÂN LOẠI DIAGNOSIS (Chỉ xét khi thuộc nhóm True Rejection) ---
                # Kiểm tra xem loại lỗi và giá trị phoneme thực tế nói ra có trùng khớp với model dự đoán không
                if trans_status == out_status and trans_val == out_val:
                    self.metrics_counts["CD"] += 1  # Transcript giống Output -> Correct Diagnosis
                else:
                    self.metrics_counts["DE"] += 1  # Transcript khác Output -> Diagnosis Error

    def compute_final_metrics(self) -> Dict[str, float]:
        mc = self.metrics_counts
        ac = self.asr_counts

        # Phân rã dữ liệu đếm
        ta, fa, tr, fr = mc["TA"], mc["FA"], mc["TR"], mc["FR"]
        cd, de = mc["CD"], mc["DE"]

        # 1. Tính toán các metric phân loại cơ bản
        total_detection = ta + fa + tr + fr
        det_accuracy = (ta + tr) / total_detection if total_detection > 0 else 0.0
        precision = tr / (tr + fr) if (tr + fr) > 0 else 0.0
        recall = tr / (tr + fa) if (tr + fa) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 2. Tính toán FRR và FAR theo logic phân phối chuẩn của MDD
        frr = fr / (ta + fr) if (ta + fr) > 0 else 0.0
        far = fa / (tr + fa) if (tr + fa) > 0 else 0.0

        # 3. Tính toán DER dựa trên công thức bạn yêu cầu: cd / (cd + de)
        der = de / (cd + de) if (cd + de) > 0 else 0.0

        # 4. Tính toán PER cho nhánh ASR
        per = (ac["substitution"] + ac["deletion"] + ac["insertion"]) / ac["total_canonical"] if ac["total_canonical"] > 0 else 0.0

        return {
            "Detection_Accuracy": det_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "FRR": frr,
            "FAR": far,
            "DER": der,
            "PER": per
        }






def evaluate(model, iterator, device, criterion_ctc, criterion_nll, w2v2, lmd=0.5):
    model.eval()
    test_loss = 0
    evaluator = MDDEvaluator()
    print("Evaluating")
    test_bar = tqdm(enumerate(iterator), total=len(iterator), desc="Evaluating")
    with torch.no_grad():
        for batch_idx, (waveform, transcript, canonical, input_length, label_length, error_gt) in test_bar:
            waveform = waveform.to(device)
            transcript = transcript.to(device)
            canonical = canonical.to(device)
            input_length = input_length.to(device)
            label_length = label_length.to(device)
            error_gt = error_gt.to(device)

            ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length,pad_idx=pad_idx)
            ctc_probs_loss = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T,B,C) — chỉ dùng cho CTCLoss
            ctc_probs_decode = F.log_softmax(ctc_logits, dim=-1)  # (B,T,C) — dùng để decode từng sample
            T = ctc_logits.shape[1]

            input_length = torch.clamp(input_length // 320, max=T)

            CTC_BLANK_ID = 0
            PAD_ID = 1

            current_batch_size = ctc_logits.size(0)

            for i in range(current_batch_size):
                single_sample_probs = ctc_probs_decode[i]  # (T, C) — đúng chuỗi thời gian của sample i
                single_target_transcript = transcript[i].squeeze().cpu().tolist()
                single_canon = canonical[i].squeeze().cpu().tolist()

                phonemes, pred_ID, argmax_path, best_probs = ctc_greedy_decode(
                    single_sample_probs, idx2p=idx2p, blank_token=CTC_BLANK_ID
                )

                # 2. Lọc Output mô hình: Bỏ Blank (0)
                clean_output = [int(x) for x in pred_ID if int(x) != CTC_BLANK_ID]

                # 3. Lọc Nhãn gốc: Bỏ PAD (1)
                clean_transcript = [int(x) for x in single_target_transcript if int(x) != PAD_ID]
                clean_canon = [int(x) for x in single_canon if int(x) != PAD_ID]

                evaluator.update_sample(
                    canonical=clean_canon,
                    transcript=clean_transcript,
                    output=clean_output
                )
                if batch_idx == 0 and i == 0:
                    print("--- 🔬 DEBUG SHAPE BEFORE EVALUATOR ---")
                    print(f"Canon Type/Shape: {type(single_canon)} | Content: {single_canon[:5]}")
                    print(
                        f"Transcript Type/Shape: {type(single_target_transcript)} | Content: {single_target_transcript[:5]}")
                    print(f"Output Type/Shape: {type(clean_output)} | Content: {clean_output[:5]}")

            ctc_loss = criterion_ctc(ctc_probs_loss, transcript, input_length, label_length)
            nll_loss = criterion_nll(
                nll_logits.reshape(-1, 2),
                error_gt.reshape(-1)
            )
            loss = ctc_loss*lmd + (1-lmd)*nll_loss
            test_loss += loss.item()
            test_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

            metrics = evaluator.compute_final_metrics()

    return test_loss / max(1, len(iterator)), metrics

import gc

def main():
    train_path = cfg.TRAIN_PATH
    dev_path = cfg.DEV_PATH
    test_path = cfg.TEST_PATH

    train_dataset = pei_dataset(train_path)
    dev_dataset = pei_dataset(dev_path)
    test_dataset = pei_dataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=2,
        persistent_workers=False,
        collate_fn=collate_fn,
        batch_size=6
    )

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
        collate_fn=collate_fn,
        batch_size=6
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
        collate_fn=collate_fn,
        batch_size=6
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w2v2 = wav2vec2(target_layers=[9])
    w2v2 = w2v2.to(device)

    model = pei().to(device)
    print(list(model.named_children()))
    for name, module in model.named_children():
        if name != "w2v2":
            module.apply(weight_init)

    criterion_ctc = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True
    )

    criterion_nll = nn.CrossEntropyLoss(ignore_index=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    patience = cfg.PATIENCE
    counter = 0
    best_loss = float("inf")

    checkpoint_dir = cfg.CHECKPOINT_DIR + "/checkpoint.pth"
    try:
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["best_loss"]
        print(f"best loss : {best_loss}")

        print("Loaded model")
    except FileNotFoundError:
        print("Checkpoint does not exist, training from scratch")

    except (pickle.UnpicklingError, AttributeError) as e:
        # Triggered by security blocks (weights_only=True) or corrupted files
        print(
            f"Serialization Error: File is corrupted or contains untrusted code. {e}"
        )

    except RuntimeError as e:
        # Triggered by mismatched layer shapes or device (CPU/GPU) conflicts
        print(f"PyTorch Runtime Error: {e}")

    except Exception as e:
        # Fallback catch-all for any other unexpected issues
        print(f"An unexpected error occurred: {e}")



    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")
        
        train_loss = train(
            model,
            train_loader,
            device,
            criterion_ctc,
            criterion_nll,
            optimizer,
            w2v2
        )

        dev_loss = validate(
            model,
            dev_loader,
            device,
            criterion_ctc,
            criterion_nll,
            w2v2
        )

        visualize_random_samples(model, dev_dataset, device, idx2p, w2v2)

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f}"
        )

        if dev_loss < best_loss:
            print(f"Model improved by {best_loss - dev_loss}")
            best_loss = dev_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler=None,
                loss=best_loss,
                epoch=epoch,
                checkpoint_dir=cfg.CHECKPOINT_DIR
            )

            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("EARLY STOPPING")
            break
        
        gc.collect()
        torch.cuda.empty_cache()

    visualize_random_samples(model, test_dataset, device, idx2p, w2v2)
    test_loss, metrics = evaluate(model,test_loader, device, criterion_ctc, criterion_nll, w2v2)
    print(metrics)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
"""


import os
import gc
import ast
import math
import pickle
import random
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm


from pyctcdecode import build_ctcdecoder


from Model import pei
from Wav2vec2 import wav2vec2
from cfg import cfg

import numpy as np
from collections import defaultdict



# ============================================================
# 1. VOCAB — tách khỏi model, load theo từng thực nghiệm
# ============================================================

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    p2idx = vocab["p2idx"]
    idx2p = vocab["idx2p"]
    return p2idx, idx2p


def resolve_special_ids(p2idx, blank_id=0):

    if "[PAD]" not in p2idx:
        raise KeyError("Vocab không có key '[PAD]' — kiểm tra lại file vocab.pkl")
    pad_id = p2idx["[PAD]"]
    return pad_id, blank_id


# ============================================================
# 2. DATASET
# ============================================================

def load_audio_without_torchaudio(path, target_sr=16000):
    sr, waveform = wavfile.read(path + ".wav")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if waveform.dtype.kind in {"i", "u"}:
        max_value = max(abs(float(waveform.min())), abs(float(waveform.max())), 1.0)
        waveform = waveform.astype("float32") / max_value
    else:
        waveform = waveform.astype("float32")

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        waveform = resample_poly(waveform, up, down).astype("float32")
        sr = target_sr

    waveform = torch.from_numpy(waveform).unsqueeze(0)

    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        waveform = torch.nan_to_num(waveform)

    return waveform, sr


def parse_sequence(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


class pei_dataset(Dataset):
    def __init__(self, data_path, processor=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.processor = processor or Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-100h"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = sample["Path"]
        waveform, sr = load_audio_without_torchaudio(path, target_sr=16000)

        waveform_np = waveform.squeeze(0).numpy()
        inputs = self.processor(waveform_np, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)

        transcript = torch.tensor(parse_sequence(sample["Transcript"]), dtype=torch.long)
        canonical = torch.tensor(parse_sequence(sample["Canonical"]), dtype=torch.long)
        error = torch.tensor(parse_sequence(sample["Error_GT"]), dtype=torch.long)

        return input_values, transcript, canonical, error


def make_collate_fn(pad_idx):

    def collate_fn(batch):
        input_values = [item[0] for item in batch]
        transcript = [item[1] for item in batch]
        canonical = [item[2] for item in batch]
        error_gt = [item[3] for item in batch]

        input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)
        trans_padded = pad_sequence(transcript, batch_first=True, padding_value=pad_idx)
        canonical_padded = pad_sequence(canonical, batch_first=True, padding_value=pad_idx)
        error_gt_padded = pad_sequence(error_gt, batch_first=True, padding_value=2)

        input_lengths = torch.LongTensor([s.size(0) for s in input_values])
        label_lengths = torch.LongTensor([l.size(0) for l in transcript])

        return (
            input_values_padded,
            trans_padded,
            canonical_padded,
            input_lengths,
            label_lengths,
            error_gt_padded
        )
    return collate_fn


# ============================================================
# 3. KHỞI TẠO TRỌNG SỐ
# ============================================================

def weight_init(module):
    if isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_pei_weights(model):

    model.apply(weight_init)


# ============================================================
# 4. DECODE — greedy (dự phòng) + beam search (đúng như paper)
# ============================================================

class PhonemeDecoder:
    def __init__(self, idx2p, p2idx, blank_id=0, beam_size=10):
        self.idx2p = idx2p
        self.p2idx = p2idx
        self.blank_id = blank_id
        self.beam_size = beam_size

        # Xây dựng danh sách nhãn chuẩn theo đúng thứ tự index từ vocab
        # Giống hệt cách repo gốc trích xuất danh sách nhãn cho pyctcdecode
        max_idx = max(idx2p.keys()) if idx2p else 0
        labels_list = [idx2p.get(i, "") for i in range(max_idx + 1)]

        # Đảm bảo blank token hoặc các token đặc biệt không làm lệch index của pyctcdecode
        # Khởi tạo ctc_decoder từ thư viện chuẩn
        self.decoder_ctc = build_ctcdecoder(labels=labels_list)

    def decode(self, log_probs_single):
        """
        Nhận vào log_probs_single dạng tensor hoặc numpy array (T, V)
        hoặc (B, T, V) và thực thi giải mã bằng pyctcdecode.
        """
        if torch.is_tensor(log_probs_single):
            log_probs = log_probs_single.detach().cpu().numpy()
        else:
            log_probs = log_probs_single

        # Nếu tensor truyền vào có dạng (1, T, V), bóp chiều batch lại
        if log_probs.ndim == 3:
            log_probs = log_probs.squeeze(0)

        # pyctcdecode yêu cầu đầu vào là xác suất dạng log-probabilities hoặc probabilities
        # tuỳ cấu hình, ở đây ta truyền trực tiếp numpy array qua decoder chuẩn.
        decoded_text = self.decoder_ctc.decode(log_probs)

        # Nếu từ decoder trả về dạng chuỗi văn bản/âm vị, ta map ngược lại thành list token IDs
        # để MDDEvaluator xử lý đúng định dạng đầu vào.
        # Hoặc nếu bạn muốn tách chuỗi theo khoảng trắng để lấy lại list token:
        pred_tokens = []
        for token_str in decoded_text.strip().split():
            if token_str in self.p2idx:
                pred_tokens.append(self.p2idx[token_str])
            else:
                # Xử lý trường hợp ký tự lạ nếu có
                if "[UNK]" in self.p2idx:
                    pred_tokens.append(self.p2idx["[UNK]"])

        return pred_tokens


# ============================================================
# 5. MDD EVALUATOR (giữ nguyên logic — đã xác nhận đúng định nghĩa paper)
# ============================================================

class MDDEvaluator:
    def __init__(self):
        self.metrics_counts = {"TA": 0, "FA": 0, "TR": 0, "FR": 0, "CD": 0, "DE": 0}
        self.asr_counts = {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0, "total_canonical": 0}

    def _to_list(self, seq):
        if isinstance(seq, torch.Tensor):
            return seq.detach().cpu().tolist()
        return list(seq)

    def calculate_edit_distance(self, hyp, ref):
        hyp_list = self._to_list(hyp)
        ref_list = self._to_list(ref)

        n, m = len(hyp_list), len(ref_list)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if hyp_list[i - 1] == ref_list[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )

        i, j = n, m
        sub_cnt, del_cnt, ins_cnt, cor_cnt = 0, 0, 0, 0
        ref_mapping = [None] * m

        while i > 0 or j > 0:
            if i > 0 and j > 0 and hyp_list[i - 1] == ref_list[j - 1]:
                cor_cnt += 1
                ref_mapping[j - 1] = ("correct", hyp_list[i - 1])
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
                del_cnt += 1
                ref_mapping[j - 1] = ("del", -1)
                j -= 1
            elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
                ins_cnt += 1
                i -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                sub_cnt += 1
                ref_mapping[j - 1] = ("sub", hyp_list[i - 1])
                i -= 1
                j -= 1

        return cor_cnt, sub_cnt, del_cnt, ins_cnt, ref_mapping

    def update_sample(self, canonical, transcript, output):
        canonical_list = self._to_list(canonical)

        _, _, _, _, trans_map = self.calculate_edit_distance(transcript, canonical)
        o_cor, o_sub, o_del, o_ins, out_map = self.calculate_edit_distance(output, canonical)

        self.asr_counts["correct"] += o_cor
        self.asr_counts["substitution"] += o_sub
        self.asr_counts["deletion"] += o_del
        self.asr_counts["insertion"] += o_ins
        self.asr_counts["total_canonical"] += len(canonical_list)

        for j in range(len(canonical_list)):
            trans_status, trans_val = trans_map[j]
            out_status, out_val = out_map[j]

            is_trans_correct = (trans_status == "correct")
            is_out_correct = (out_status == "correct")

            if is_trans_correct and is_out_correct:
                self.metrics_counts["TA"] += 1
            elif not is_trans_correct and is_out_correct:
                self.metrics_counts["FA"] += 1
            elif is_trans_correct and not is_out_correct:
                self.metrics_counts["FR"] += 1
            else:
                self.metrics_counts["TR"] += 1
                if trans_status == out_status and trans_val == out_val:
                    self.metrics_counts["CD"] += 1
                else:
                    self.metrics_counts["DE"] += 1

    def compute_final_metrics(self):
        mc = self.metrics_counts
        ac = self.asr_counts

        ta, fa, tr, fr = mc["TA"], mc["FA"], mc["TR"], mc["FR"]
        cd, de = mc["CD"], mc["DE"]

        total_detection = ta + fa + tr + fr
        det_accuracy = (ta + tr) / total_detection if total_detection > 0 else 0.0
        precision = tr / (tr + fr) if (tr + fr) > 0 else 0.0
        recall = tr / (tr + fa) if (tr + fa) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        frr = fr / (ta + fr) if (ta + fr) > 0 else 0.0
        far = fa / (tr + fa) if (tr + fa) > 0 else 0.0
        der = de / (cd + de) if (cd + de) > 0 else 0.0
        per = (ac["substitution"] + ac["deletion"] + ac["insertion"]) / ac["total_canonical"] if ac["total_canonical"] > 0 else 0.0

        return {
            "Detection_Accuracy": det_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "FRR": frr,
            "FAR": far,
            "DER": der,
            "PER": per,
        }


# ============================================================
# 6. TRAIN / VALIDATE / EVALUATE
# ============================================================

def train(model, iterator, device, criterion_ctc, criterion_nll, optimizer, w2v2, pad_id, lmd=0.5, max_norm=2.0):
    model.train()
    train_loss = 0
    train_bar = tqdm(enumerate(iterator), total=len(iterator), desc="training")
    for _, (waveform, transcript, canonical, input_length, label_length, error_gt) in train_bar:
        waveform = waveform.to(device)
        transcript = transcript.to(device)
        canonical = canonical.to(device)
        input_length = input_length.to(device)
        label_length = label_length.to(device)
        error_gt = error_gt.to(device)

        ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length, pad_idx=pad_id)

        batch_size, max_T, _ = ctc_logits.shape

        ctc_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T,B,C) cho CTCLoss
        T = ctc_logits.shape[1]
        #input_length_ctc = torch.clamp(input_length // 320, max=T)
        #input_length = torch.full(size=(ctc_logits.shape[1],), fill_value=ctc_logits.shape[0], dtype=torch.long,device=device)

        input_length_ctc = torch.full(size=(batch_size,), fill_value=max_T, dtype=torch.long, device=device)

        #nll_logits = F.log_softmax(nll_logits, dim=2)
        ctc_loss = criterion_ctc(ctc_probs, transcript, input_length_ctc, label_length.view(-1).long())
        nll_loss = criterion_nll(nll_logits.reshape(-1, 2), error_gt.reshape(-1))
        loss = ctc_loss * lmd + (1 - lmd) * nll_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

    return train_loss / max(1, len(iterator))


def validate(model, iterator, device, criterion_ctc, criterion_nll, w2v2, pad_id, lmd=0.5):
    model.eval()
    dev_loss = 0
    dev_bar = tqdm(enumerate(iterator), total=len(iterator), desc="validating")
    with torch.no_grad():
        for _, (waveform, transcript, canonical, input_length, label_length, error_gt) in dev_bar:
            waveform = waveform.to(device)
            transcript = transcript.to(device)
            canonical = canonical.to(device)
            input_length = input_length.to(device)
            label_length = label_length.to(device)
            error_gt = error_gt.to(device)

            ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length, pad_idx=pad_id)

            batch_size, max_T, _ = ctc_logits.shape
            
            ctc_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            T = ctc_logits.shape[1]
            #input_length_ctc = torch.clamp(input_length // 320, max=T)
            #input_length = torch.full(size=(ctc_logits.shape[1],), fill_value=ctc_logits.shape[0], dtype=torch.long,device=device)
            #nll_logits = F.log_softmax(nll_logits, dim=2)
            input_length_ctc = torch.full(size=(batch_size,), fill_value=max_T, dtype=torch.long, device=device)

            ctc_loss = criterion_ctc(ctc_probs, transcript, input_length_ctc, label_length.view(-1).long())
            nll_loss = criterion_nll(nll_logits.reshape(-1, 2), error_gt.reshape(-1))
            loss = ctc_loss * lmd + (1 - lmd) * nll_loss

            dev_loss += loss.item()
            dev_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

    return dev_loss / max(1, len(iterator))


def evaluate(model, iterator, device, criterion_ctc, criterion_nll, w2v2, decoder, pad_id, lmd=0.5):
    
    model.eval()
    evaluator = MDDEvaluator()
    test_loss = 0
    test_bar = tqdm(enumerate(iterator), total=len(iterator), desc="Evaluating")
    with torch.no_grad():
        for batch_idx, (waveform, transcript, canonical, input_length, label_length, error_gt) in test_bar:
            waveform = waveform.to(device)
            transcript = transcript.to(device)
            canonical = canonical.to(device)
            input_length = input_length.to(device)
            label_length = label_length.to(device)
            error_gt = error_gt.to(device)

            ctc_logits, nll_logits = model(waveform, canonical, w2v2, input_lengths=input_length, pad_idx=pad_id)

            batch_size, max_T, _ = ctc_logits.shape
            input_length_ctc = torch.full(size=(batch_size,), fill_value=max_T, dtype=torch.long, device=device)

            # Tách riêng 2 bản — tránh lặp lại bug transpose đã gặp trước đây
            ctc_probs_loss = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)   # (T,B,C) cho CTCLoss
            ctc_probs_decode = F.log_softmax(ctc_logits, dim=-1)                 # (B,T,C) cho decode từng sample

            T = ctc_logits.shape[1]
            #input_length_ctc = torch.clamp(input_length // 320, max=T)
            #input_length = torch.full(size=(ctc_logits.shape[1],), fill_value=ctc_logits.shape[0], dtype=torch.long,device=device)

            batch_size = ctc_logits.size(0)
            for i in range(batch_size):
                single_probs = ctc_probs_decode[i]
                single_transcript = [t for t in transcript[i].cpu().tolist() if t != pad_id]
                single_canon = [t for t in canonical[i].cpu().tolist() if t != pad_id]

                clean_output = decoder.decode(single_probs)

                evaluator.update_sample(
                    canonical=single_canon,
                    transcript=single_transcript,
                    output=clean_output
                )
            #nll_logits = F.log_softmax(nll_logits, dim=2)
            ctc_loss = criterion_ctc(ctc_probs_loss, transcript, input_length_ctc, label_length.view(-1).long())
            nll_loss = criterion_nll(nll_logits.reshape(-1, 2), error_gt.reshape(-1))
            loss = ctc_loss * lmd + (1 - lmd) * nll_loss

            test_loss += loss.item()
            test_bar.set_postfix(loss=f"{loss.item():.3f}", ctc=f"{ctc_loss.item():.3f}", nll=f"{nll_loss.item():.3f}", refresh=False)

    metrics = evaluator.compute_final_metrics()
    print("Raw detection counts:", evaluator.metrics_counts)
    print("Raw ASR counts:", evaluator.asr_counts)
    return test_loss / max(1, len(iterator)), metrics


def visualize_random_samples(model, dataset, device, idx2p, w2v2, decoder, pad_id, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n--- VISUALIZING {num_samples} RANDOM SAMPLES ---")

    with torch.no_grad():
        for idx in indices:
            waveform, transcript, canon, _ = dataset[idx]
            waveform = waveform.unsqueeze(0).to(device)
            canon = canon.unsqueeze(0).to(device)
            input_lengths = torch.LongTensor([waveform.size(1)]).to(device)

            ctc_logits, _ = model(waveform, canon, w2v2, input_lengths=input_lengths, pad_idx=pad_id)
            ctc_probs = F.log_softmax(ctc_logits, dim=-1)  # (1, T, C)

            pred_ids = decoder.decode(ctc_probs[0])
            clean_pred = [str(idx2p[i]) for i in pred_ids]
            clean_target = [str(idx2p[t.item()]) for t in transcript if t.item() != pad_id]

            print(f"Sample index: {idx}")
            print(f" > Target: {' '.join(clean_target)}")
            print(f" > Output: {' '.join(clean_pred)}")
            print("-" * 30)


def save_checkpoint(model, optimizer, scheduler=None, loss=None, epoch=None,
                     checkpoint_dir="checkpoint", filename="checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': loss,
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print("Saved Checkpoint")


# ============================================================
# 7. MỘT THỰC NGHIỆM HOÀN CHỈNH — độc lập hoàn toàn với các thực nghiệm khác
# ============================================================

def run_experiment(
    name: str,
    vocab_path: str,
    train_path: str,
    dev_path: str,
    test_path: str,
    checkpoint_dir: str,
    use_beam_decode: bool = True,
    device: torch.device = None,
):

    print(f"\n{'=' * 60}\nTHỰC NGHIỆM: {name}\n{'=' * 60}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- vocab ---
    p2idx, idx2p = load_vocab(vocab_path)
    pad_id, blank_id = resolve_special_ids(p2idx, blank_id=0)
    phone = len(p2idx)
    print(f"Vocab size: {phone} | pad_id={pad_id} (-> '{idx2p.get(pad_id, '?')}') "
          f"| blank_id={blank_id} (-> '{idx2p.get(blank_id, '?')}')")
    print(">> Kiểm tra 2 dòng trên cho khớp ý định của bạn trước khi train!")

    # --- data ---
    shared_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
    train_dataset = pei_dataset(train_path, processor=shared_processor)
    dev_dataset = pei_dataset(dev_path, processor=shared_processor)
    test_dataset = pei_dataset(test_path, processor=shared_processor)

    collate_fn = make_collate_fn(pad_id)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0,
                               persistent_workers=False, collate_fn=collate_fn, batch_size=16)
    dev_loader = DataLoader(dev_dataset, shuffle=False, num_workers=0,
                             persistent_workers=False, collate_fn=collate_fn, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0,
                              persistent_workers=False, collate_fn=collate_fn, batch_size=16)

    # --- model / w2v2 (w2v2 đứng ngoài model, không train, không bị weight_init đụng tới) ---
    w2v2 = wav2vec2(target_layers=[9]).to(device)
    model = pei(phone=phone, pad_idx=pad_id).to(device)
    init_pei_weights(model)
    
    criterion_ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    criterion_nll = nn.CrossEntropyLoss(ignore_index = 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.DECAY)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = None

    #decoder = NativeCTCBeamSearch(idx2p, p2idx, blank_id=blank_id, beam_size=10)
    decoder = PhonemeDecoder(
        idx2p=idx2p,
        p2idx=p2idx,
        blank_id=blank_id,
        beam_size=10
    )

    best_loss = float("inf")
    counter = 0

    for epoch in range(cfg.EPOCHS):
        print(f"\n[{name}] Epoch {epoch + 1}/{cfg.EPOCHS}")

        train_loss = train(model, train_loader, device, criterion_ctc, criterion_nll, optimizer, w2v2, pad_id)
        dev_loss = validate(model, dev_loader, device, criterion_ctc, criterion_nll, w2v2, pad_id)
        #scheduler.step()

        print(f"[{name}] Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

        if dev_loss < best_loss:
            print(f"[{name}] Model improved by {best_loss - dev_loss:.4f}")
            best_loss = dev_loss
            save_checkpoint(model, optimizer, scheduler=None, loss=best_loss,
                             epoch=epoch, checkpoint_dir=checkpoint_dir)
            counter = 0
        else:
            counter += 1

        if counter >= cfg.PATIENCE:
            print(f"[{name}] EARLY STOPPING")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # --- load lại checkpoint tốt nhất trước khi visualize/eval cuối cùng ---
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[{name}] Đã load lại checkpoint tốt nhất (epoch {checkpoint['epoch']}, "
              f"dev_loss={checkpoint['best_loss']:.4f})")

    visualize_random_samples(model, test_dataset, device, idx2p, w2v2, decoder, pad_id)

    test_loss, metrics = evaluate(model, test_loader, device, criterion_ctc, criterion_nll, w2v2, decoder, pad_id)

    print(f"\n[{name}] === KẾT QUẢ TEST ===")
    print(f"[{name}] Test Loss: {test_loss:.4f}")
    for k, v in metrics.items():
        print(f"[{name}] {k}: {v * 100:.2f}%")

    return {"name": name, "test_loss": test_loss, **metrics}


# ============================================================
# 8. MAIN — 3 thực nghiệm vocab, cùng 1 model, cùng 1 pipeline
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 thực nghiệm: mỗi cái có vocab_path RIÊNG và bộ CSV RIÊNG (vì Canonical/
    # Transcript/Error_GT trong CSV được encode theo đúng vocab tương ứng —
    # không được lẫn CSV của vocab này với vocab kia).
    # Điền đúng đường dẫn CSV thật của bạn vào 6 dòng train/dev/test dưới đây.
    experiments = [
        dict(
            name="old_vocab",
            vocab_path="vocab_old.pkl",
            train_path=cfg.TRAIN_PATH_OLD,
            dev_path=cfg.DEV_PATH_OLD,
            test_path=cfg.TEST_PATH_OLD,
            checkpoint_dir=cfg.CHECKPOINT_DIR_OLD,
        ),
        dict(
            name="new_vocab",
            vocab_path="vocab_new.pkl",
            train_path=cfg.TRAIN_PATH_NEW,
            dev_path=cfg.DEV_PATH_NEW,
            test_path=cfg.TEST_PATH_NEW,
            checkpoint_dir=cfg.CHECKPOINT_DIR_NEW,
        ),
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(device=device, **exp)
        all_results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\nSO SÁNH OLD VOCAB vs NEW VOCAB\n{'=' * 60}")
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    df.to_csv("vocab_experiments_comparison.csv", index=False)
    print("\nĐã lưu bảng so sánh vào vocab_experiments_comparison.csv")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    import multiprocessing
    multiprocessing.freeze_support()
    main()

