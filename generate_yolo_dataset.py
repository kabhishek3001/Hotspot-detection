import os
import random

from PIL import Image, ImageDraw

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
NUM_IMAGES_PER_SHAPE = 5000
MIN_SHAPE_SIZE = 50
MAX_SHAPE_SIZE = 250

SHAPE_CLASSES = {
    "Rectangle": 0,
    "Square": 1,
    "Circle": 2,
    "Triangle": 3
}

BACKGROUND_COLORS = [
    (200, 200, 200), (220, 220, 220), (240, 240, 240), 
    (173, 216, 230), (135, 206, 250), (176, 224, 230), 
    (144, 238, 144), (152, 251, 152), (60, 179, 113), 
    (255, 228, 196), (255, 239, 213), (250, 235, 215), 
    (255, 192, 203), (255, 182, 193), (221, 160, 221), 
]

SHAPE_COLORS = [
    (255, 0, 0), 
    (0, 0, 255), 
    (0, 128, 0), 
    (255, 165, 0), 
    (128, 0, 128), 
    (255, 20, 147), 
    (0, 0, 0), 
    (70, 130, 180), 
    (218, 165, 32), 
    (165, 42, 42), 
    (0, 200, 0), 
    (200, 0, 200), 
]

BBOX_PADDING_PERCENTAGE = 0.07  # 7% padding around the tight bounding box

def create_directories_for_shape(base_path: str, shape_name: str) -> tuple:
    """
    Creates image and label directories for a given shape.
    
    Args:
        base_path (str): Base path for the dataset.
        shape_name (str): Name of the shape (e.g., "Rectangle").
    
    Returns: 
        tuple: A tuple containing the image and label directories.
    """
    
    images_dir = os.path.join(base_path, "images", shape_name)
    labels_dir = os.path.join(base_path, "labels", shape_name)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    return images_dir, labels_dir

def get_random_color_pair(bg_colors: list, shape_colors: list) -> tuple:
    """
    Selects a random background and shape color, ensuring they are not identical.
    
    Args:
        bg_colors (list): List of background colors.
        shape_colors (list): List of shape colors.
    
    Returns:
        tuple: A tuple containing the selected background and shape colors.
    """

    bg_color = random.choice(bg_colors)
    shape_color = random.choice(shape_colors)
    
    while shape_color == bg_color: # Simple check, might need more sophisticated contrast check for real-world
        shape_color = random.choice(shape_colors)
    
    return bg_color, shape_color

def get_triangle_area(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculates the area of a triangle given its three vertices.
    
    Args:
        p1 (tuple): Coordinates of the first vertex (x, y).
        p2 (tuple): Coordinates of the second vertex (x, y).
        p3 (tuple): Coordinates of the third vertex (x, y).
    
    Returns:
        float: Area of the triangle.
    """
    
    # 0.5 * |(x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))|
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def calculate_yolo_annotation(tight_bbox: tuple, class_id: int, img_w: int, img_h: int, padding_percentage: float=0.07) -> str:
    """
    Calculates YOLO annotation string from a tight bounding box.
    
    Args:
        tight_bbox (tuple): Tight bounding box coordinates (x_min, y_min, x_max, y_max).
        class_id (int): Class ID of the shape.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
        padding_percentage (float): Percentage of padding to add around the bounding box.
    
    Returns:
        str: YOLO annotation string.
    """

    tight_x_min, tight_y_min, tight_x_max, tight_y_max = tight_bbox

    tight_w = tight_x_max - tight_x_min
    tight_h = tight_y_max - tight_y_min

    pad_w = tight_w * padding_percentage
    pad_h = tight_h * padding_percentage

    enlarged_x_min = tight_x_min - pad_w
    enlarged_y_min = tight_y_min - pad_h
    enlarged_x_max = tight_x_max + pad_w
    enlarged_y_max = tight_y_max + pad_h

    final_x_min = max(0, enlarged_x_min)
    final_y_min = max(0, enlarged_y_min)
    final_x_max = min(img_w, enlarged_x_max)
    final_y_max = min(img_h, enlarged_y_max)
    
    if final_x_max <= final_x_min:
        final_x_max = final_x_min + 1 if final_x_min < img_w else img_w
        if final_x_max > img_w: final_x_max = img_w
        if final_x_min >= final_x_max : final_x_min = final_x_max -1 if final_x_max >0 else 0
            
    if final_y_max <= final_y_min:
        final_y_max = final_y_min + 1 if final_y_min < img_h else img_h
        if final_y_max > img_h: final_y_max = img_h
        if final_y_min >= final_y_max: final_y_min = final_y_max-1 if final_y_max >0 else 0

    bbox_abs_center_x = (final_x_min + final_x_max) / 2.0
    bbox_abs_center_y = (final_y_min + final_y_max) / 2.0
    bbox_abs_width = final_x_max - final_x_min
    bbox_abs_height = final_y_max - final_y_min

    x_center_norm = bbox_abs_center_x / img_w
    y_center_norm = bbox_abs_center_y / img_h
    width_norm = bbox_abs_width / img_w
    height_norm = bbox_abs_height / img_h

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def generate_rectangle(draw: ImageDraw, shape_color: tuple) -> tuple:
    """
    Generates a rectangle, draws it, and returns its tight bounding box.

    Args:
        draw (ImageDraw): ImageDraw object for drawing on the image.
        shape_color (tuple): RGB color tuple for the shape.

    Returns:
        tuple: Tight bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    
    width = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    height = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    
    x1 = random.randint(0, IMAGE_WIDTH - width)
    y1 = random.randint(0, IMAGE_HEIGHT - height)
    
    x2 = x1 + width
    y2 = y1 + height
    
    draw.rectangle([(x1, y1), (x2, y2)], fill=shape_color, outline=None)
    return (x1, y1, x2, y2)

def generate_square(draw: ImageDraw, shape_color: tuple) -> tuple:
    """
    Generates a square, draws it, and returns its tight bounding box.
    
    Args:
        draw (ImageDraw): ImageDraw object for drawing on the image.
        shape_color (tuple): RGB color tuple for the shape.
    
    Returns: 
        tuple: Tight bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    
    side = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    
    x1 = random.randint(0, IMAGE_WIDTH - side)
    y1 = random.randint(0, IMAGE_HEIGHT - side)
    
    x2 = x1 + side
    y2 = y1 + side
    
    draw.rectangle([(x1, y1), (x2, y2)], fill=shape_color, outline=None)
    return (x1, y1, x2, y2)

def generate_circle(draw: ImageDraw, shape_color: tuple) -> tuple:
    """
    Generates a circle, draws it, and returns its tight bounding box.
    
    Args: 
        draw (ImageDraw): ImageDraw object for drawing on the image.
        shape_color (tuple): RGB color tuple for the shape.
    
    Returns: 
        tuple: Tight bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    
    radius = random.randint(MIN_SHAPE_SIZE // 2, MAX_SHAPE_SIZE // 2)
    
    cx = random.randint(radius, IMAGE_WIDTH - radius)
    cy = random.randint(radius, IMAGE_HEIGHT - radius)
    
    x1 = cx - radius
    y1 = cy - radius
    x2 = cx + radius
    y2 = cy + radius
    
    draw.ellipse([(x1, y1), (x2, y2)], fill=shape_color, outline=None)
    return (x1, y1, x2, y2)

def generate_triangle(draw: ImageDraw, shape_color: tuple) -> tuple:
    """
    Generates a random triangle, draws it, and returns its tight bounding box.
    
    Args: 
        draw (ImageDraw): ImageDraw object for drawing on the image.
        shape_color (tuple): RGB color tuple for the shape.
    
    Returns: 
        tuple: Tight bounding box coordinates (x_min, y_min, x_max, y_max).
    """

    min_area = (MIN_SHAPE_SIZE * MIN_SHAPE_SIZE) / 10.0
    points = []
    attempts = 0
    
    while attempts < 50:
        tri_bbox_w = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        tri_bbox_h = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        
        tri_origin_x = random.randint(0, IMAGE_WIDTH - tri_bbox_w)
        tri_origin_y = random.randint(0, IMAGE_HEIGHT - tri_bbox_h)

        temp_points = []
        for _ in range(3):
            pt_x = random.randint(tri_origin_x, tri_origin_x + tri_bbox_w)
            pt_y = random.randint(tri_origin_y, tri_origin_y + tri_bbox_h)
            temp_points.append((pt_x, pt_y))
        
        area = get_triangle_area(temp_points[0], temp_points[1], temp_points[2])
        if area > min_area:
            points = temp_points
            break
        
        attempts += 1
    
    if not points:
        base = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        height = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        
        tl_x = random.randint(0, IMAGE_WIDTH - base)
        tl_y = random.randint(0, IMAGE_HEIGHT - height)

        p1 = (tl_x + base // 2, tl_y)
        p2 = (tl_x, tl_y + height)
        p3 = (tl_x + base, tl_y + height)
        
        points = [p1, p2, p3]

    draw.polygon(points, fill=shape_color, outline=None)
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

if __name__ == "__main__":
    dataset_base_path = "dataset_2d_shapes"
    os.makedirs(dataset_base_path, exist_ok=True)

    shape_generators = {
        "Rectangle": generate_rectangle,
        "Square": generate_square,
        "Circle": generate_circle,
        "Triangle": generate_triangle
    }

    for shape_name, class_id in SHAPE_CLASSES.items():
        print(f"Generating dataset for: {shape_name} (Class ID: {class_id})")
        images_dir, labels_dir = create_directories_for_shape(dataset_base_path, shape_name)
        
        shape_generator_func = shape_generators[shape_name]

        for i in range(NUM_IMAGES_PER_SHAPE):
            bg_color, shape_color_val = get_random_color_pair(BACKGROUND_COLORS, SHAPE_COLORS)
            
            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), bg_color)
            draw = ImageDraw.Draw(image)
            
            tight_bbox = shape_generator_func(draw, shape_color_val)
            yolo_annotation_str = calculate_yolo_annotation(
                tight_bbox, class_id, IMAGE_WIDTH, IMAGE_HEIGHT, BBOX_PADDING_PERCENTAGE
            )
            
            image_filename = f"{shape_name.lower()}_{i+1:04d}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            image.save(image_path)
            
            label_filename = f"{shape_name.lower()}_{i+1:04d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, "w") as f:
                f.write(yolo_annotation_str)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{NUM_IMAGES_PER_SHAPE} images for {shape_name}")

    print(f"\nDataset generation complete! Images and labels are saved in '{dataset_base_path}' directory.")
    
    print("Dataset structure:")
    print(f"{dataset_base_path}/")
    
    print("  images/")
    for shape_name in SHAPE_CLASSES.keys():
        print(f"    {shape_name}/")
    
    print("  labels/")
    for shape_name in SHAPE_CLASSES.keys():
        print(f"    {shape_name}/")