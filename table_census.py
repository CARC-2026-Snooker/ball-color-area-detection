import cv2
import numpy as np

def count_balls_on_field(image_path):
    # 1. Load Image & Convert to HSV
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    output_img = img.copy()

    # 2. Color Thresholds (Same as your previous code)
    lower_red1 = np.array([0, 60, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 60, 20])
    upper_red2 = np.array([180, 255, 255])
    
    lower_blue = np.array([60, 40, 20])
    upper_blue = np.array([150, 255, 255])

    # 3. Create Global Masks
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- THE NEW PART: FINDING CONTOURS (BLOBS) ---
    # cv2.findContours looks for continuous groupings of white pixels in the mask
    # RETR_EXTERNAL means we only want the outside outline of the ball
    # CHAIN_APPROX_SIMPLE saves memory by only storing the corner points of the blob
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum size of a blob to be considered a real ball (ignores tiny noise pixels)
    MIN_AREA = 120
    
    red_count = 0
    blue_count = 0

    # 4. Count Red Balls
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            red_count += 1
            # Draw a green outline around the detected ball
            cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
            
            # Find the center of this blob to put a label on it
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(output_img, f"R{red_count}", (cx - 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Count Blue Balls
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            blue_count += 1
            # Draw a yellow outline around the detected ball
            cv2.drawContours(output_img, [cnt], -1, (0, 255, 255), 2)
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(output_img, f"B{blue_count}", (cx - 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Print the final census
    print("=== FIELD CENSUS ===")
    print(f"Total Red Balls Found : {red_count}")
    print(f"Total Blue Balls Found: {blue_count}")
    print("====================")

    cv2.imshow("Field Census", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return red_count, blue_count

# Run the test
if __name__ == "__main__":
    count_balls_on_field("camera_images\img_21.jpg")