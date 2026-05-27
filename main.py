import cv2
import numpy as np

def detect_balls_in_zones(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'. Check the file path.")
        return

    # 2. Define your Regions of Interest (ROIs)
    zones = {
        # zones for images/test4.png
        "A": (150, 100, 30, 30),
        "B": (250, 100, 30, 30),
        "C": (240, 270, 40, 40),
        "D": (180, 180, 60, 60),
    }

    # 3. Convert the whole image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. Define Color Thresholds
    # RED: Red wraps around the 0-180 mark in OpenCV, so we need two ranges
    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 20])
    upper_red2 = np.array([180, 255, 255])
    
    # BLUE
    lower_blue = np.array([86, 50, 20])
    upper_blue = np.array([140, 255, 255])

    # WHITE (any hue, low saturation, high value)
    # lower_white = np.array([0, 0, 100]) # enabled
    lower_white = np.array([180, 75, 255]) # disabled
    upper_white = np.array([180, 75, 255])

    # --- ADJUSTABLE PERCENTAGE THRESHOLD ---
    # 0.15 means at least 15% of the zone's total area must contain the color to register as a ball
    PERCENT_THRESHOLD = 0.10

    # --- GENERATE GLOBAL MASKS FOR DEBUGGING ---
    # Creating masks for the whole image makes tuning your HSV values significantly easier
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    global_mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    global_mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    global_mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Prepare Image 1: Pure masks combined (Black background, Red mask = Red, Blue mask = Blue)
    debug_masks_img = np.zeros_like(img)
    debug_masks_img[global_mask_red > 0] = (0, 0, 255)   # Red in BGR
    debug_masks_img[global_mask_blue > 0] = (255, 0, 0)  # Blue in BGR
    debug_masks_img[global_mask_white > 0] = (255, 255, 255)  # White in BGR

    # Prepare Image 2: Translucent overlay on top of the original image
    mask_color_layer = np.zeros_like(img)
    mask_color_layer[global_mask_red > 0] = (0, 0, 255)
    mask_color_layer[global_mask_blue > 0] = (255, 0, 0)
    mask_color_layer[global_mask_white > 0] = (255, 255, 255)
    # Blend 70% original image with 30% mask color layer
    debug_overlay_img = cv2.addWeighted(img, 0.7, mask_color_layer, 0.3, 0)

    # Create a fresh copy of the original image to draw final detection results onto
    output_img = img.copy()

    # 5. Process Each Zone
    for zone_name, (x, y, w, h) in zones.items():
        # Slice out the zone's area directly from our pre-calculated global masks
        roi_red = global_mask_red[y:y+h, x:x+w]
        roi_blue = global_mask_blue[y:y+h, x:x+w]
        roi_white = global_mask_white[y:y+h, x:x+w]

        # Count how many mask pixels are active inside this specific bounding box
        red_pixel_count = cv2.countNonZero(roi_red)
        blue_pixel_count = cv2.countNonZero(roi_blue)
        white_pixel_count = cv2.countNonZero(roi_white)

        # Calculate the total pixel area of this specific zone box
        zone_area = w * h

        # Convert counts into percentages (0.0 to 1.0)
        red_percent = red_pixel_count / zone_area
        blue_percent = blue_pixel_count / zone_area
        white_percent = white_pixel_count / zone_area

        # 6. Logic to determine what is in the zone using percentages
        status = "Empty"
        color_to_draw = (255, 255, 255) 
        ball_center = None # Variable to store our center (if found)

        if red_percent > PERCENT_THRESHOLD and red_percent > blue_percent and red_percent > white_percent:
            status = f"Red Ball ({red_percent:.1%})"
            color_to_draw = (0, 0, 255) 
            
            # Find center of the RED blob
            y_indices, x_indices = np.where(roi_red > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))
            
        elif blue_percent > PERCENT_THRESHOLD and blue_percent > white_percent:
            status = f"Blue Ball ({blue_percent:.1%})"
            color_to_draw = (255, 0, 0) 
            
            # Find center of the BLUE blob
            y_indices, x_indices = np.where(roi_blue > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))
        elif white_percent > PERCENT_THRESHOLD:
            status = f"White Ball ({white_percent:.1%})"
            color_to_draw = (255, 255, 255) 
            
            # Find center of the WHITE blob
            y_indices, x_indices = np.where(roi_white > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))

        # 7. Draw the box, text, AND the center dot
        for target_img in [output_img, debug_masks_img, debug_overlay_img]:
            cv2.rectangle(target_img, (x, y), (x + w, y + h), color_to_draw, 2)
            cv2.putText(target_img, f"{zone_name}: {status}", (x, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_to_draw, 1, cv2.LINE_AA)
            
            # IF a ball was detected, draw a small target circle at its exact center pixel
            if ball_center is not None:
                # Draw a small filled circle (radius 3 pixels) in green to stand out
                cv2.circle(target_img, ball_center, 2, (0, 255, 255), -1) 
        
        # Print telemetry data including the center pixel coordinates
        center_str = f" Center: {ball_center}" if ball_center else ""
        print(f"Zone {zone_name} -> Decided: {status}{center_str}")
        
        # Print telemetry data to the console
        print(f"Zone {zone_name} -> Red: {red_percent:.1%}, Blue: {blue_percent:.1%}, White: {white_percent:.1%}. Decided: {status}")

    # Save the files straight to your working directory
    cv2.imwrite("debug_masks_combined.png", debug_masks_img)
    cv2.imwrite("debug_overlay.png", debug_overlay_img)
    print("\n[SUCCESS] Saved debug images to your directory: 'debug_masks_combined.png' & 'debug_overlay.png'")

    # 8. Show the final windows
    cv2.imshow("1. Snooker Bot Vision - Results", output_img)
    cv2.imshow("2. Debug - Combined Pure Masks", debug_masks_img)
    cv2.imshow("3. Debug - Translucent Overlay", debug_overlay_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- RUN THE TEST ---
if __name__ == "__main__":
    detect_balls_in_zones("images/test4.png")