from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Проверка CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a model
    model = YOLO(r"D:\Viacheslav\PycharmProjects\YOLO_segmentation\runs\segment\train3\weights\best.pt")

    # Perform inference on an image
    results = model('rgb.png', conf=0.15, device=0)
    results[0].show()