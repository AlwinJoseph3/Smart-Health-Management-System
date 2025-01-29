from ultralytics import YOLO

def main():

    model = YOLO("yolov8l")

    config_file_path = "D:/1_MAIN PROJECT/Image Processing/MRI/code/data.yaml"
    project = "D:/1_MAIN PROJECT/Image Processing Models"
    experiment = "MRI model"
    batch_size = 9

    result = model.train(data=config_file_path,
                         epochs=50,
                         project=project,
                         name=experiment,
                         batch=batch_size,
                         device="0",
                         patience=50,
                         imgsz=350,
                         verbose=True,
                         val=True)
    
if __name__ == "__main__":
    main()

