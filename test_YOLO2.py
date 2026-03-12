from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Проверка CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a model
    model = YOLO("yolo26n-seg.pt")  # load a pretrained segmentation model

    # Train the model
    results = model.train(
        data="package-seg.yaml",
        epochs=100,
        imgsz=640,
        device=0
    )

    # Validate the model
    results = model.val()

    # Perform inference on an image
    results = model('rgb.png', conf=0.15, device=0)
    results[0].show()