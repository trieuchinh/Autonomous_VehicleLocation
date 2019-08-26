# Autonomous_VehicleLocation
The purpose of this program is to determine position of car on the road. Position could be Left, Right and Middle
<br/>This program is a small part of auto-driving as well
<br/>There are few steps that expose to this problem
- Color segmentation
- Convert to gray scale
- Smoothing with Gaussian filter
- Determine region of interest to remove redundant data
- Do detection on edge using CannyEdge detector
- Do line segmentation using HougLineP algorithm. Compared to traditional HoughLine. HougLine with probability is better
- Check wheter lane is either solid or dash lane at each side of the car. Using clustering method combined with standard deviation 
- Based on type of lane(Dash, Solid), You're able to distinguish LEFT, RIGHT, MIDDLE

<br />This program was written in Python language. To run this program, some python packages are required such as opencv, sklearn. numpy.
<br />[To install opencv python](https://pypi.org/project/opencv-python/)
<br />[To install numpy or sklearn](https://scipy.org/install.html)

<br />For more: 
<br />Visit [Reduction Algorithm](https://trieuchinhblog.blogspot.com/2017/09/parallel-reduction-algorithm.html)
<br />Visit [Contour Algorithm](https://trieuchinhblog.blogspot.com/2017/09/96-normal-0-false-false-false-en-us-ja.html)
