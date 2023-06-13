import numpy as np
import cv2

# Checkerboard parameters
num_corners_x = 8  # Number of interior corners along the x-axis
num_corners_y = 6  # Number of interior corners along the y-axis
square_size = 1.0  # Size of each square in the checkerboard (in any unit)

# Create arrays to store 3D and 2D coordinates of checkerboard corners
object_points = []  # 3D coordinates of checkerboard corners in the world coordinate system
image_points = []  # 2D coordinates of checkerboard corners in the image

# Generate coordinates of the corners in the world coordinate system
object_points_grid = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
object_points_grid[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2) * square_size

# Load images and find checkerboard corners
image_paths = ['check_board.png']  # Replace with the paths to your checkerboard images

for path in image_paths:
    # Read the image
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

    # If corners are found, add them to the lists
    if ret:
        object_points.append(object_points_grid)
        image_points.append(corners)

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Print the obtained calibration parameters
print("Camera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", distortion_coeffs)
