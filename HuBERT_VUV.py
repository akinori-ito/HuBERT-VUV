import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, AutoFeatureExtractor
import librosa
import numpy as np

# --- 設定 ---
SAMPLING_RATE = 16000 # HuBERTは通常16kHzを想定
FRAME_RATE_MS = 20    # 分類を行うフレーム長 (HuBERTの出力と一致させる)
MODEL_NAME = "facebook/hubert-base-ls960"

class CustomSadDataset(Dataset):
    def __init__(self, data_list, feature_extractor):
        # data_list: [(音声ファイルパス, 10msラベルファイルパス), ...]
        self.data_list = data_list
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, label_path = self.data_list[idx]

        # 1. 音声のロードと特徴抽出
        speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
        # feature_extractorは、生の音声波形を入力として受け取る
        inputs = self.feature_extractor(
            speech, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.squeeze(0) # [input_seq_len]

        # 2. ラベルのロード
        target_labels_20ms = np.load(label_path)
                
        # 3. HuBERTの出力長に合わせたラベル長の調整
        # HuBERTの特徴量抽出ストライドは、通常16kHzで320サンプル(20ms)
        # ここでは、入力サンプルの長さからHuBERTの出力長を近似的に推定
        # (正確な式は (L_in - 400) // 320 + 1 などだが、ここでは簡略化)
        target_len_approx = input_values.shape[0] // 320 
        
        # ラベルをターゲット長に合わせて切り詰める（HuBERTの出力長に合わせる）
        if len(target_labels_20ms) > target_len_approx:
            target_labels_20ms = target_labels_20ms[:target_len_approx]
        
        target_labels = torch.tensor(target_labels_20ms, dtype=torch.long)
        
        return {"input_values": input_values, "labels": target_labels}

# --- ダミーデータのリスト作成例 (実際には独自のデータパスに置き換えてください) ---
# train_data_list = [
#     ("path/to/audio1.wav", "path/to/label1_10ms.npy"),
#     # ...
# ]
class HubertForSad(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME):
        super().__init__()
        # 1. HuBERTモデルのロード
        self.hubert = HubertModel.from_pretrained(model_name)
        
        # 2. 分類ヘッドの定義
        config = self.hubert.config
        #print(config)
        hidden_size = config.hidden_size # 通常768
        
        self.dropout = nn.Dropout(config.final_dropout)
        # HuBERTの出力次元からクラス数への全結合層
        self.classifier = nn.Linear(hidden_size, num_classes) 

    def forward(self, input_values, attention_mask=None, labels=None):
        # HuBERTによる特徴量の抽出
        # output.last_hidden_state: [batch_size, seq_len, hidden_size]
        output = self.hubert(input_values, attention_mask=attention_mask)
        
        # 分類ヘッド
        hidden_states = output.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states) # [B, T, C]

        loss = None
        if labels is not None:
            # 損失関数: パディングされたラベル(-100)を自動で無視するCrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # ロジット: (B, T, C) -> (B*T, C)
            logits_flat = logits.view(-1, self.classifier.out_features) 
            
            # ラベル: (B, T) -> (B*T)
            labels_flat = labels.view(-1)
            
            loss = loss_fct(logits_flat, labels_flat)

        return (loss, logits) if loss is not None else logits

# --- DataLoaderのためのカスタムcollate_fn ---
def custom_collate_fn(batch):
    # input_values (音声波形)をパディング
    input_values = [item['input_values'] for item in batch]
    input_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    
    # labels (フレームラベル)をパディング。パディング値-100はロス計算で無視される
    labels = [item['labels'] for item in batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) 

    return {'input_values': input_padded, 'labels': labels_padded}
