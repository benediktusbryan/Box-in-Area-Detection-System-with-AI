#Box-in-Area-Detection-System-with-AI

This system detect Box in 3x3 Areas. Furthermore this system can be integrated with AGV to carry the box.
Using yolov7 for the AI. https://github.com/WongKinYiu/yolov7

##Model
Download model1.pt then copy to main directory
https://drive.google.com/file/d/12xcC1BRS2F1yNi9RrE6jnbL7Fe0_g9vU/view?usp=sharing

##Training
For training yolov7. Use this Google Collab https://colab.research.google.com/drive/1jdXmSPX4GBePFuJJYKiIOMQo-iTjqzci?usp=sharing

##Result
https://user-images.githubusercontent.com/52401633/186632539-b845d01d-f96a-41bf-add8-2b64a9741deb.mp4
![image](https://user-images.githubusercontent.com/52401633/186638287-2885f699-564c-4936-afdc-eda304b0b545.png)

Blue region is defined as empty area. Red region is defined as filled area.
The algorithm to decide area is using center weight of the detected bounding box coordinate and compared to the defined 3x3 area.

![image_2022Aug22_141544](https://user-images.githubusercontent.com/52401633/186634041-20f74908-a64c-41dd-802c-2ca70ea1b200.jpg)
yolov7 is the basic algorithm. The confidence level of the detected object are between 0.44 to 0.85 (maximum 1.00)

![confusion_matrix](https://user-images.githubusercontent.com/52401633/186634589-b880ad3d-0196-4a55-abbc-e91289696cb2.png)
Confusion matrix shows testing result that 100% accuracy for detecting box but 0% accuracy for detecting person

![results](https://user-images.githubusercontent.com/52401633/186635098-acd733e9-80d5-470f-ad12-475909b6f7ad.png)
Precision from this model is 1.00 (maximum 1.00). The detected bounding box is always true. But Recall is 0.5 (maximum 1.00). From all boxes in the testing images, the model successfully detect. But for all person in the testing images, the model totally failed to detect.

##Run
Open terminal

cd Box-in-Area-Detection-System-with-AI
python ./main.py

Open browser

For streaming the camera

http://localhost:8000/mjpg

For get result

http://localhost:8000/hasil

Example Response body

{
  "area 0": "0",
  "area 1": "1",
  "area 2": "0",
  "area 3": "1",
  "area 4": "0",
  "area 5": "0",
  "area 6": "0",
  "area 7": "0",
  "area 8": "0"
}
