import os
import glob

def count_classes(labels_dir):
    class_counts = {0: 0, 1: 0, 2: 0}
    for file_path in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    class_label = int(line.split()[0])
                    class_counts[class_label] += 1
                except (ValueError, IndexError):
                    print(f"Warning: Skipping invalid line in file: {file_path}")
                    continue
    return class_counts

labels_dir = "075_ObjectDetection_Yolo7/yolov7/train/labels"
class_counts = count_classes(labels_dir)
print(f"Class counts: {class_counts}")
