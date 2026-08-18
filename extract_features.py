import torch
# import transformers # wav2vec2
# import timm # VideoMAE

def extract_audio_features(wav_path, model_name="facebook/wav2vec2-base"):
    """
    Skeleton cho việc trích xuất đặc trưng Audio bằng wav2vec2.
    
    Args:
        wav_path (str): Đường dẫn tới file audio.
        
    Returns:
        audio_features (Tensor): (seq_len, 768)
    """
    print(f"Trích xuất audio cho {wav_path} bằng {model_name}...")
    # TODO: Khởi tạo processor và model wav2vec2
    # processor = Wav2Vec2Processor.from_pretrained(model_name)
    # model = Wav2Vec2Model.from_pretrained(model_name)
    
    # Giả lập kết quả
    seq_len = 50
    return torch.randn(seq_len, 768)

def extract_visual_features(video_path, model_name="MCG-NJU/videomae-base"):
    """
    Skeleton cho việc trích xuất đặc trưng Visual bằng VideoMAE hoặc OpenFace.
    
    Args:
        video_path (str): Đường dẫn tới file video.
        
    Returns:
        visual_features (Tensor): (seq_len, 256) (tuỳ chỉnh lại số chiều cho phù hợp)
    """
    print(f"Trích xuất visual cho {video_path} bằng {model_name}...")
    # TODO: Cài đặt logic cho VideoMAE / OpenFace
    
    # Giả lập kết quả
    seq_len = 50
    return torch.randn(seq_len, 256)

def main():
    print("Script dùng để trích xuất đặc trưng offline (Task 2).")
    print("Sẽ lưu kết quả thành file .pkl hoặc .npy để Dataloader đọc.")
    
    # Ví dụ
    # data_dir = "path/to/iemocap"
    # for file in os.listdir(data_dir):
    #     a_feat = extract_audio_features(file_audio)
    #     v_feat = extract_visual_features(file_video)
    #     save_to_disk(a_feat, v_feat)

if __name__ == '__main__':
    main()
