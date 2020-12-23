# Thesis Project
## Development of Neural-Network based Star Identification System

### What's in this repo?
1. My first project using the built in matching method from OpenCV Brute Force Matching with SIFT Descriptors and Ratio Test.
1. Testing of Star Pattern Recognition using Convolutional Neural Networks.

### Short Background
  The development of artificial intelligence and computer vision throughout the years have been accelerated exponentially 
by the growth of the hardware performance such as the memory storage. Most star trackers rely on huge onboard computers to
store the data of the star catalogue and rely on brute force algorithms to match the star image to the star catalogue to determine
a satellite's attitude. Brute Force Algorithm also has the tendency to have a longer processing time as it needed to search through
the whole catalogue. It also has a trade off between the accuracy and the processing time which is also a weakness of its operation.
Convolutional Neural Network architectures as a form of object identification process have been proven succesful.
The use of CNNs would minimize the memory storage as CNNs that an onboard computer need to have and thus could reduce cost so lower budget
satellites would be able to use this sensor as a means of attitude determination. The use of CNN would also create a non dependent processing time
after it has been trained. In short, this is my attempt to mimic the human brain to produce accurate attitude determination system for
spacecrafts.

### Main Libraries Used
* OpenCV <br/>
![OpenCV_Logo](https://user-images.githubusercontent.com/32363208/97399385-a94cdf00-191f-11eb-825c-4be7c90f9b3d.png)
* Tensorflow <br/>
![tensorflow-logo-AE5100E55E-seeklogo com](https://user-images.githubusercontent.com/32363208/97399448-c84b7100-191f-11eb-90ac-902ee352c543.png)
