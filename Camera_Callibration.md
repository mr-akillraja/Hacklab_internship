# Camera Callibration
- Methods to find a Camera's internal and External parameters.
## Internal and External parameters
- Internal Parameters (Intrinsic parameters)
    - Focal Length(fx,fy) : 
        - Represents the distance between the camera's image plane and the lens,measured in pixels.
        - It determines the field of view and the magnification of the camera.
    - Principal Point(cx,cy) :
        - It is the intersection point of the camera's optical axis with the image plane.
        - represents the center of the image.
    - Lens Distortion Cofficients(k1,k2,p1,p2,k3) :
        - These cofficients capture the imprefections in the camera lens causing distortions in the image.
        - distortions can be radial (k1,k2,k3) or tangential (p1,p2) distortions.

- External Parameters (Extrinsic parameters)
    - Rotations Matrix(R) : 
        - Describes the rotation of the camera with respect to the world coordinate system.
        - The rotation matrix represents how the camera is oriented in three-dimensional space.
    - Translation(t) :
        - Represents the translation or the displacement of the camera with respect to the world coordinate system. 
        - Translation vector denotes the position of the camera in three dimensional space.

## Topics 
# Linear Model

### Forward Imaging Model: 3D to 2D ###
- World coordinates(Xw ;3D) --> Camera coordinates(Xc ;3D) --> Image Coordinates(Xi ;2D)
- Perpective Projection :
    - 3 Point in the world coordinate is projected onto a 2D point in the image plane of the camera
    - (X,Y,Z) -> the 3D coordinates of a point in the world coordinate system.
    - (x,y) -> the 2D coordinates of the corresponding of the pont in the camera's image plane.
    - Equation is :
        x =  (f*X)/Z + cx
        y = (f*y)/z + cy
        where,
        f -> focal length of the camera
        (cx,cy) -> principal point,which is the coordinates of the image center in pixels.
        Z -> depth or the distance of the 3D point from the camera's image plane.
        (x,y) -> represents the pixel coordinates of the projected point in the  camera's image.

### Image plane to Image Sensor ###
- process of mapping the image plane to the image sensor involves transforming the 2D image coordinates in the image plane to the corresponding coordinates on the image sensor.
- Steps involved : 
    - Determine the Image Plane:
        - Image Plane is the two-dimensional space where the image is formed.
        - It is perpendicular to the camera's optical axis and intersects with it at the focal point.
        - The size of the image plane is defined by the dimensions of the camera's sensor.
    - Calculate the Image Plane Coordinates :
        - Given a point(x,y) in the image plane where (0,0) represents the top-left corner,calculate the corresponding image plane coordinates.
    - Apply Perspective Projection :
        - Convert the image plane coordinates to normalized image plane coordinates.
    - Convert to Image Sensor Coordinates :
        - map the normalized image plane coordinates to the image sensor coordinates.
        - This steps involved scaling the normalized coordinates by the dimensions of the image sensor and considering the orientation of the sensor.
    - Equation :
        - x_sensor = (x_plane/pixel_size) + (image_width/2)
        - y_sensor =  (y_plane/pixel_size) + (image_height/2)
        where 
            -(x_plane,y_plane) represents the coordinates of a point in the image plane.
            - (x_sensor,y_sensor) represents the corresponding coordinates of the point on the image sensor.
            - pixel_size represents the size of a pixel on the image sensor.
            - image_width,image_height are the dimensions of the image sensors.
