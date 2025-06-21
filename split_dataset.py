import os
import shutil
import random
import glob

SOURCE_IMAGES_DIR = os.path.join("hotspot_dataset", "images", "train")
SOURCE_LABELS_DIR = os.path.join("hotspot_dataset", "labels", "train")

OUTPUT_DATA_DIR = "yolo_hotspot_data"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def get_corresponding_label_file(image_path: str, labels_dir: str) -> str:
    """
    Gets the corresponding label file path for a given image file path.
    
    Args:
        image_path (str): Path to the image file.
        labels_dir (str): Directory containing label files.
    
    Returns:
        str: Path to the corresponding label file.
    """
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, base_filename + ".txt")

def main():
    print(f"Source Images Directory: {os.path.abspath(SOURCE_IMAGES_DIR)}")
    print(f"Source Labels Directory: {os.path.abspath(SOURCE_LABELS_DIR)}")
    print(f"Output Data Directory: {os.path.abspath(OUTPUT_DATA_DIR)}")

    if not os.path.isdir(SOURCE_IMAGES_DIR):
        print(f"Error: Source images directory not found: {SOURCE_IMAGES_DIR}")
        return
    if not os.path.isdir(SOURCE_LABELS_DIR):
        print(f"Error: Source labels directory not found: {SOURCE_LABELS_DIR}")
        return

    train_images_path = os.path.join(OUTPUT_DATA_DIR, "images", "train")
    train_labels_path = os.path.join(OUTPUT_DATA_DIR, "labels", "train")
    val_images_path = os.path.join(OUTPUT_DATA_DIR, "images", "val")
    val_labels_path = os.path.join(OUTPUT_DATA_DIR, "labels", "val")

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    image_files = glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.png"))
    if not image_files:
        print(f"Error: No image files found in {SOURCE_IMAGES_DIR}. Ensure images were generated correctly.")
        return
        
    print(f"Found {len(image_files)} total images.")

    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    split_index = int(len(image_files) * TRAIN_RATIO)

    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Splitting into {len(train_files)} training images and {len(val_files)} validation images.")

    def copy_files(file_list: list, dest_img_dir: str, dest_lbl_dir: str, source_lbl_dir: str) -> int:
        """
        Copy image and label files to the destination directories.
        
        Args:
            file_list (list): List of image file paths to copy.
            dest_img_dir (str): Destination directory for images.
            dest_lbl_dir (str): Destination directory for labels.
            source_lbl_dir (str): Source directory for labels.
        
        Returns:
            int: Number of successfully copied image-label pairs.
        """
        
        copied_count = 0
        for img_path in file_list:
            lbl_path = get_corresponding_label_file(img_path, source_lbl_dir)

            if not os.path.exists(lbl_path):
                print(f"Warning: Label file not found for {img_path}, skipping.")
                continue

            try:
                shutil.copy(img_path, dest_img_dir)
                shutil.copy(lbl_path, dest_lbl_dir)
                copied_count +=1

            except Exception as e:
                print(f"Error copying {img_path} or {lbl_path}: {e}")

        return copied_count

    print("\nCopying training files...")
    train_copied = copy_files(train_files, train_images_path, train_labels_path, SOURCE_LABELS_DIR)
    print(f"Copied {train_copied} training image-label pairs.")

    print("\nCopying validation files...")
    val_copied = copy_files(val_files, val_images_path, val_labels_path, SOURCE_LABELS_DIR)
    print(f"Copied {val_copied} validation image-label pairs.")

    print(f"\nDataset splitting complete. Output is in: {os.path.abspath(OUTPUT_DATA_DIR)}")
    print("Structure:")
    print(f"  {OUTPUT_DATA_DIR}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  ({train_copied} images)")
    print(f"  │   └── val/    ({val_copied} images)")
    print(f"  └── labels/")
    print(f"      ├── train/  ({train_copied} labels)")
    print(f"      └── val/    ({val_copied} labels)")

if __name__ == "__main__":
    main()