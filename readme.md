# The Heart Sounds Project

Six years ago, I watched helplessly as my dad fainted before my eyes. My mother screamed and rushed him to the hospital, where we discovered he had a severe case of mitral valve prolapse. He had to replace one of the valves in his heart and flew to New York for open-heart surgery, leaving my sister and me behind in Mumbai. He spent almost five months in recovery, and I was overwhelmed with worry. I remember the doctor telling him that if they had caught it earlier, he might not have needed the surgery at all. 

This experience inspired my project, aimed at helping millions of people like my dad to avoid similar suffering. I developed a CNN neural network based on the ResNet-18 architecture to classify spectrograms of heartbeats into five categories: aortic stenosis, mitral regurgitation, mitral valve prolapse, mitral stenosis, and normal. I trained the model on a dataset of 1,000 images, with 200 per category, using 2-second audio clips of heartbeats from Kaggle. The data was collected from the general public via the iStethoscope Pro iPhone app and from a clinical trial in hospitals using the DigiScope digital stethoscope.

To prepare the data, I normalized the audio to 1, resampled it to 16kHz, and sorted it into the five categories. I then converted each audio clip into a spectrogram and used transfer learning on the ResNet-18 model to train it with these spectrogram images. This project represents my effort to create a tool that could potentially aid in the early detection of heart conditions, sparing others from the ordeal my family went through.

image

## The Algorithm

Add an explanation of the algorithm and how it works. Make sure to include details about how the code works, what it depends on, and any other relevant info. Add images or other descriptions for your project here. 

## Running this project

1. Add steps for running this project.
2. Make sure to include any required libraries that need to be installed for your project to run.

[View a video explanation here](video link)
