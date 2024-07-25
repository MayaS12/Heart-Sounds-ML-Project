# The Heart Sounds Project

Six years ago, I watched helplessly as my dad fainted before my eyes. My mother screamed and rushed him to the hospital, where we discovered he had a severe case of mitral valve prolapse. He had to replace one of the valves in his heart and flew to New York for open-heart surgery, leaving my sister and me behind in Mumbai. He spent almost five months in recovery, and I was overwhelmed with worry. I remember the doctor telling him that if they had caught it earlier, he might not have needed the surgery at all. 

This experience inspired my project, aimed at helping millions of people like my dad to avoid similar suffering. I developed a CNN neural network based on the ResNet-18 architecture to classify spectrograms of heartbeats into five categories: aortic stenosis, mitral regurgitation, mitral valve prolapse, mitral stenosis, and normal. I trained the model on a dataset of 1,000 images, with 200 per category, using 2-second audio clips of heartbeats from Kaggle. The data was collected from the general public via the iStethoscope Pro iPhone app and from a clinical trial in hospitals using the DigiScope digital stethoscope.

To prepare the data, I normalized the audio to 1, resampled it to 16kHz, and sorted it into the five categories. I then converted each audio clip into a spectrogram and used transfer learning on the ResNet-18 model to train it with these spectrogram images. This project represents my effort to create a tool that could potentially aid in the early detection of heart conditions, sparing others from the ordeal my family went through.

image

## The Algorithm

There are 4 major parts of my algorithm: 
1. Wav to spectrogram conversion:
I converted .wav audio files to spectrogram plots using libraries like Matplotlib, Librosa, and NumPy. This process included cleaning up and normalizing the data to ensure consistency and reliability. Spectrograms visualize the frequency spectrum of the audio signals over time, making it easier to analyze and classify different heartbeat sounds.

3. Neural networks:
Neural networks are computational models inspired by the human brain's neural structure. They consist of layers of interconnected nodes, or neurons, where each connection has associated weights and biases. These networks learn to make predictions or classify data through a process called backpropagation, which adjusts the weights and biases based on the error of the output. Neural networks are widely used in various applications, such as image and speech recognition, due to their ability to model complex patterns and relationships in data.

5. Resnet 18 architecture:
ResNet-18 is a type of convolutional neural network (CNN). It consists of 18 hidden layers and introduces the concept of residual learning, where shortcut connections (or skips) allow the network to learn identity mappings. This helps to mitigate the vanishing gradient problem and enables the training of deeper networks. ResNet-18 is used in my project to effectively extract and learn features from the spectrogram images of heartbeats.

7. Transfer learning:
Transfer learning involves taking a pre-trained model on a large dataset and fine-tuning it for a specific task on a smaller, related dataset. This approach leverages the knowledge already learned by the model, reducing the amount of data and computational resources needed for training. In my project, I used transfer learning on the ResNet-18 model, fine-tuning it with the spectrogram images of heartbeats to classify them into five categories: aortic stenosis, mitral regurgitation, mitral valve prolapse, mitral stenosis, and normal. The benefits of transfer learning include improved performance, faster training times, and the ability to achieve good results with limited data.

## Running this project:
Note: To complete this project, I cleaned, normalized and converted all the audio files into spectrograms. I have included the code in the repository but to run this project on your own computer, I would suggest just using the final data provided in this repository of clean spectrograms. 

1. First, begin by setting up an SSH conection with your Jetson Nano (with jetson-inference configured) and opening a functioning terminal.
  
2. Use cd commands to change directories until you are in your jetson-inference/python/training/classification/data
   
`cd jetson-inference/python/training/classification/data`

3. Run this command to download the image dataset.

`wget https://github.com/MayaS12/Heart-Sounds-ML-Project/tree/main/heart_sounds`

4. cd back to 'classification' directory, and then cd into the 'models' directory 

`cd ..`
`cd models`

5. Run this command to download the skin cancer classification model.

`wget https://github.com/MayaS12/Heart-Sounds-ML-Project/tree/main/heart_sounds2`

6. cd back to 'classification' directory

`cd ..`

7. Use the following command to make sure that the model is on the nano. You should see a file called resnet18.onnx.

 `ls models/heart_sounds/` 

8. Set the NET and DATASET variables by running each of these commands separately

`NET=models/heart_sounds2`

`DATASET=data/heart_sounds`

9. Run this command try the model and see how it operates on an image from the test folder!! Change 'NAME HERE' to name your output file and rename 'NAME OF CATEGORY' and 'IMAGE NAME'
    
`imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/NAME OF CATEGORY/IMAGE NAME .jpg $DATASET/test_output_NAME OF CATEGORY/NAME OUTPUT.jpg`

This is an example of what your command should look like after you replace the fill-ins.

`imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/MVP/New_MVP_164.png $DATASET/test_output_MVP/test164.png`

10. Look at your results by opening the image that just saved in the 'test_output' folder! This folder should be located in jetson-inference/python/training/classification/data/heart_sounds/test_output


[View a video explanation here](video link)
