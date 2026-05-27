from ultralytics import YOLO

model = YOLO(r"C:\Users\R RAHUL\OneDrive\Desktop\container\runs\detect\train4\weights\last.pt")

model.train(
    resume=True
)