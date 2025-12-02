import mlflow
from ultralytics import YOLO
import os

# MLflow experiment tracking for YOLO training
EXPERIMENT_NAME = "anime-yolo-training"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_with_mlflow(data_yaml='dataset.yaml', epochs=50, imgsz=640, model_name='yolov8n.pt'):
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("imgsz", imgsz)
        mlflow.log_param("data_yaml", data_yaml)
        
        model = YOLO(model_name)
        results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
        
        # Log final metrics
        if os.path.exists('runs/detect/train'):
            mlflow.log_artifacts('runs/detect/train', artifact_path="training_outputs")
        
        print("Training complete. MLflow run logged.")

if __name__ == '__main__':
    train_with_mlflow()
