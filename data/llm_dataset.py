from torch.utils.data import Dataset
import numpy as np

from dataset import VibrationDataset

def rms_ac(vib: np.ndarray):
    vib = vib - vib.mean()
    return float(np.sqrt(np.mean(np.power(vib, 2))))

def order_one_channel(sig: np.ndarray, sr: float, rpm, od):
    sig = np.asarray(sig, dtype=np.float64)
    N = sig.shape[-1] # 시간 영역 샘플 개수
    if N < 8 or sr <= 0 or rpm <= 0:
        return 0.0 
    sig_ac = sig - sig.mean()
    w = np.hanning(N)
    xw = sig_ac * w

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    df = sr / N

    f_rot = rpm / 60.0
    f_target = od * f_rot # 분석하고자 하는 주파수 (order)

    bw_hz = max (2.0 * df, 0.1 * f_target)
    band_mask = np.abs(freqs - f_target) <= bw_hz 

    order_val = np.abs(X[band_mask])
    return float(order_val.max()) # order 근처를 포함한 대역의 |X|의 내의 최대값을 반환

def p2p(vib: np.ndarray):

    x = np.asarray(vib, dtype=np.float64)
    if x.size == 0:
        return 0.0
    peakTopeak = float(x.max() - x.min())
    return peakTopeak


class LLM_Dataset(Dataset):
    def __init__(self, vibration_dataset:VibrationDataset,
                include_ref= True):
        super().__init__()
        
        self.vibration_dataset = vibration_dataset
        
    def __len__(self):
        return len(self.vibration_dataset)

    def feature_extract(self, vibration:np.array, sr: float, rpm: float):
        """_summary_

        Args:
            vibration (np.array): 진동 데이터

        Returns:
            feature_dict (dict): vibration에서 추출한 특징 dict
        """
        rms_x = rms_ac(vibration[0])
        rms_y = rms_ac(vibration[1])
        
        odx_1x = order_one_channel(vibration[0], sr, rpm=rpm, od=1) # x축의 order
        odx_2x = order_one_channel(vibration[0], sr, rpm=rpm, od=2)
        odx_3x = order_one_channel(vibration[0], sr, rpm, od=3)

        ody_1x = order_one_channel(vibration[1], sr, rpm, od=1) # y축의 order
        ody_2x = order_one_channel(vibration[1], sr, rpm, od=2)
        ody_3x = order_one_channel(vibration[1], sr, rpm, od=3)

        p2p_x = p2p(vibration[0])
        p2p_y = p2p(vibration[1])

        feature_dict = {"rms_x": rms_x, "rms_y": rms_y, "order_x_1x": odx_1x, "order_x_2x": odx_2x, "order_x_3x": odx_3x,
                         "order_y_1x": ody_1x, "order_y_2x": ody_2x, "order_y_3x": ody_3x, "peak2peak_x": p2p_x, "peak2peak_y": p2p_y}
        return feature_dict
    
    def __getitem__(self, index):
        
        data_dict = self.vibration_dataset[index]

        sr = float(data_dict["x_info"]["sampling_rate"])
        rpm = float(data_dict["x_info"]["rpm"])
        
        x_feat = self.feature_extract(data_dict['x_vib'], sr=sr, rpm=rpm)
        
        ref_feat = None
        if 'ref_vib' in data_dict.keys():
            ref_sr = float(data_dict["ref_info"]["sampling_rate"])
            ref_rpm = float(data_dict["ref_info"]["rpm"])
            ref_feat = self.feature_extract(data_dict['ref_vib'], sr=ref_sr, rpm=ref_rpm)
        
        prompt = """
            LLM 에 들어갈 입력 Prompt
        """
        
        
        llm_data_dict = {
            'prompt' : prompt
        }
        
        return data_dict.update(llm_data_dict)
    
def get_llm_dataset(train_dataset, val_dataset):
    train_llm_dataset = LLM_Dataset(
        vibration_dataset=train_dataset
    )
    val_llm_dataset = LLM_Dataset(
        vibration_dataset=val_dataset
    )
    return train_llm_dataset, val_llm_dataset