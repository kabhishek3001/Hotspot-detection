import cv2
import numpy as np
import os
import random

IMG_WIDTH = 640
IMG_HEIGHT = 640
NUM_IMAGES = 5000
NUM_CIRCLES = 5
OUTPUT_DIR = "hotspot_dataset"
IMAGES_SUBDIR = os.path.join(OUTPUT_DIR, "images", "train")
LABELS_SUBDIR = os.path.join(OUTPUT_DIR, "labels", "train")
CLASS_ID = 0

HOTSPOT_PALETTE = [
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
    (255, 0, 0),      # Blue (BGR)
    (0, 0, 255),      # Red (BGR)
    (0, 255, 255)     # Yellow (BGR)
]

RADII_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2]

def generate_random_color() -> tuple:
    """
    Generates a random BGR color.
    
    Returns:
        tuple: A tuple representing a BGR color (blue, green, red).
    """
    
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def calculate_visible_bbox(cx: int, cy: int, r_outer: int, img_w: int, img_h: int) -> tuple:
    """
    Calculates the bounding box of the hotspot.
    Given the new generation logic, the hotspot is always fully within image bounds,
    so this function now simply returns the hotspot's absolute bounding box.
    
    Args:
        cx (int): Center x-coordinate of the hotspot.
        cy (int): Center y-coordinate of the hotspot.
        r_outer (int): Outer radius of the hotspot.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
    
    Returns:
        tuple: A tuple representing the bounding box (xmin, ymin, xmax, ymax).
    """
    x_min = cx - r_outer
    y_min = cy - r_outer
    x_max = cx + r_outer
    y_max = cy + r_outer

    # The max/min clamps from the original version are no longer strictly needed
    # for ensuring visibility, as cx, cy, r_outer generation now guarantees this.
    # However, keeping them doesn't hurt and provides an extra layer of theoretical safety.
    # For a truly "totally in image" scenario, they will effectively return the calculated x_min/max etc.
    return int(max(0, x_min)), int(max(0, y_min)), int(min(img_w, x_max)), int(min(img_h, y_max))

def to_yolo_format(bbox: tuple, img_w: int, img_h: int) -> tuple:
    """
    Converts an absolute bbox (xmin, ymin, xmax, ymax) to YOLO format.
    
    Args:
        bbox (tuple): A tuple representing the absolute bounding box.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
    
    Returns:
        tuple: A tuple representing the YOLO bounding box.
    """
    
    vis_x_min, vis_y_min, vis_x_max, vis_y_max = bbox

    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_center = (vis_x_min + vis_x_max) / 2.0
    y_center = (vis_y_min + vis_y_max) / 2.0
    width = vis_x_max - vis_x_min
    height = vis_y_max - vis_y_min

    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh

    return CLASS_ID, x_center_norm, y_center_norm, width_norm, height_norm

def apply_perspective_transform(image: np.ndarray, max_shift_ratio: float=0.1) -> np.ndarray:
    """
    Applies a random perspective transform to the image.
    
    Args:
        image (np.ndarray): The input image.
        max_shift_ratio (float): Maximum shift ratio for the perspective transform.
    
    Returns:
        np.ndarray: The transformed image.
    """
    
    rows, cols, _ = image.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])

    max_shift_x = cols * max_shift_ratio
    max_shift_y = rows * max_shift_ratio

    dx1 = random.uniform(-max_shift_x, max_shift_x)
    dy1 = random.uniform(-max_shift_y, max_shift_y)
    dx2 = random.uniform(-max_shift_x, max_shift_x)
    dy2 = random.uniform(-max_shift_y, max_shift_y)
    dx3 = random.uniform(-max_shift_x, max_shift_x)
    dy3 = random.uniform(-max_shift_y, max_shift_y)
    dx4 = random.uniform(-max_shift_x, max_shift_x)
    dy4 = random.uniform(-max_shift_y, max_shift_y)

    pts2 = np.float32([
        [dx1, dy1],
        [cols - 1 + dx2, dy2],
        [dx3, rows - 1 + dy3],
        [cols - 1 + dx4, rows - 1 + dy4]
    ])
    
    pts2[:,0] = np.clip(pts2[:,0], -cols*0.2, cols*1.2)
    pts2[:,1] = np.clip(pts2[:,1], -rows*0.2, rows*1.2)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return transformed_image

def apply_blur(image: np.ndarray, max_kernel_size: int=7) -> np.ndarray:
    """
    Applies Gaussian blur with a random odd kernel size.
    
    Args:
        image (np.ndarray): The input image.
        max_kernel_size (int): Maximum kernel size for the Gaussian blur.
    
    Returns:
        np.ndarray: The blurred image.
    """
    
    if max_kernel_size < 1:
        return image
    
    kernel_val = random.randrange(1, max_kernel_size + 1, 2)
    blurred_image = cv2.GaussianBlur(image, (kernel_val, kernel_val), 0)
    
    return blurred_image

def main():
    os.makedirs(IMAGES_SUBDIR, exist_ok=True)
    os.makedirs(LABELS_SUBDIR, exist_ok=True)

    print(f"Generating {NUM_IMAGES} images for YOLO dataset...")

    generated_count = 0
    # The loop limit (NUM_IMAGES * 2) is a safety net in the original code
    # to account for some images being skipped. With the new logic ensuring objects
    # are always in frame, fewer skips related to visibility are expected.
    for i in range(NUM_IMAGES * 2): 
        if generated_count >= NUM_IMAGES:
            break
        
        bg_color = generate_random_color()
        image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), bg_color, dtype=np.uint8)

        # --- MODIFICATION START ---
        # 1. Determine the maximum radius that can fit entirely within the image.
        max_possible_r_outer = min(IMG_WIDTH, IMG_HEIGHT) // 2
        
        # 2. Set a minimum radius. Ensure it's at least 1 pixel.
        min_r_outer = max(1, int(0.1 * min(IMG_WIDTH, IMG_HEIGHT)))

        # 3. Adjust min_r_outer if it's greater than max_possible_r_outer (e.g., for very small image dimensions
        #    where even the "minimum" chosen radius might be too large to fit).
        if min_r_outer > max_possible_r_outer:
            min_r_outer = max_possible_r_outer # Fallback to max possible radius to ensure a valid range
            
        # 4. Select the outer radius (r_outer) within this valid range.
        r_outer = random.randint(min_r_outer, max_possible_r_outer)
        
        # 5. Calculate the valid ranges for the center (cx, cy) to ensure the entire circle
        #    (with radius r_outer) stays within image bounds.
        #    The center must be at least 'r_outer' distance from each edge.
        cx_min = r_outer
        cx_max = IMG_WIDTH - r_outer
        cy_min = r_outer
        cy_max = IMG_HEIGHT - r_outer

        # Due to how r_outer is selected (r_outer <= min(IMG_WIDTH, IMG_HEIGHT) // 2),
        # cx_min will always be less than or equal to cx_max, and similarly for cy.
        # This ensures random.randint has a valid range.
        cx = random.randint(cx_min, cx_max)
        cy = random.randint(cy_min, cy_max)
        # --- MODIFICATION END ---

        actual_radii = [int(factor * r_outer) for factor in RADII_FACTORS]
        # Filter out radii that became 0 after integer conversion
        actual_radii = [r for r in actual_radii if r > 0]
        
        # This check might still cause a skip if r_outer is so small that
        # too many scaled radii become 0, but is less likely with the new min_r_outer.
        if len(actual_radii) < NUM_CIRCLES:
            continue

        current_hotspot_colors = random.sample(HOTSPOT_PALETTE, len(HOTSPOT_PALETTE))
        for k in range(NUM_CIRCLES):
            if k < len(actual_radii): # Ensure we don't try to access index out of bounds
                radius = actual_radii[k]
                color = current_hotspot_colors[k % len(current_hotspot_colors)]
                cv2.circle(image, (cx, cy), radius, color, -1)

        # The hotspot is now guaranteed to be fully visible and contained within the image.
        visible_bbox = calculate_visible_bbox(cx, cy, r_outer, IMG_WIDTH, IMG_HEIGHT)
        
        # The previous checks for `visible_bbox is None` and `min_visible_dim_fraction`
        # are no longer necessary for ensuring the hotspot is in frame, as the generation
        # logic already guarantees this. They are removed for clarity.

        # Apply augmentations AFTER bounding box calculation.
        # Note: Perspective transform can still push parts of the circle out of bounds.
        # If strict "totally in image" is required *after* transform, the bbox would need
        # to be recalculated after the transform, which is more complex.
        # This modification addresses the initial placement of the hotspot.
        if random.random() < 0.3:
            image = apply_perspective_transform(image, max_shift_ratio=0.08)
        
        if random.random() < 0.4:
            image = apply_blur(image, max_kernel_size=5)

        yolo_data = to_yolo_format(visible_bbox, IMG_WIDTH, IMG_HEIGHT)
        # Still good to check if normalized dimensions are valid, though highly unlikely to fail now.
        if not yolo_data or yolo_data[3] <= 0 or yolo_data[4] <= 0:
            continue

        img_filename = f"hotspot_{generated_count:05d}.png"
        label_filename = f"hotspot_{generated_count:05d}.txt"

        cv2.imwrite(os.path.join(IMAGES_SUBDIR, img_filename), image)
        with open(os.path.join(LABELS_SUBDIR, label_filename), "w") as f:
            f.write(f"{yolo_data[0]} {yolo_data[1]:.6f} {yolo_data[2]:.6f} {yolo_data[3]:.6f} {yolo_data[4]:.6f}\n")
        
        generated_count += 1
        if generated_count % 100 == 0:
            print(f"Generated {generated_count}/{NUM_IMAGES} images...")

    if generated_count < NUM_IMAGES:
        print(f"Warning: Only generated {generated_count} images out of {NUM_IMAGES} requested. This might occur if parameters are extremely constrained, causing many skips due to `actual_radii` length.")
    else:
        print(f"Successfully generated {generated_count} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()