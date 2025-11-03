import torch
from transformers import AutoFeatureExtractor
import HuBERT_VUV as hub
import soundfile as sf
import librosa

class VUVmodel:
    def __init__(self,model_path,model_name, num_classes, sampling_rate):
        """保存されたモデルの重みをロードする関数"""
        self.model = hub.HubertForSad(num_classes=num_classes)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)    
        self.model.to(self.device)
        self.model.eval() # 推論モードに設定
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = sampling_rate
    def predict(self,wav_file_path):
        # 1. 音声のロードと前処理
        # soundfileでロード (wav_dataはnumpy array)
        wav_data, sr = sf.read(wav_file_path) 
    
        if sr != self.sampling_rate:
            print(f"注意: サンプリングレート {sr}Hz を {self.samplint_rate}Hz にリサンプリングします。")
            wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=self.samplint_rate)
            sr = self.samplintg_rate
        
        # 音声データをHuBERTの入力形式に変換
        inputs = self.feature_extractor(wav_data, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        # 2. HuBERTモデルによる予測
        with torch.no_grad():
            logits = self.model(input_values=input_values) # [1, Seq_len, 3]

        # 最も確率の高いクラスを取得
        predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy() # [Seq_len]
        return predictions

if __name__ == "__main__":
    SAMPLING_RATE = 16000 
    MODEL_NAME = "facebook/hubert-base-ls960"
    MODEL_FILE = "hubert_sad_20ms_model.pth"
    NUM_CLASSES = 3

    model = VUVmodel(MODEL_FILE, MODEL_NAME, NUM_CLASSES, SAMPLING_RATE)
    predictions = model.predict("test.wav")
    print(predictions)