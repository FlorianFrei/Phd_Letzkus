Written on 18.11.2025, in the Future, code is (maybe) more updated on my Github  https://github.com/FlorianFrei/Phd_Letzkus/tree/main/pupil_anlysis/dlc 

requirements:
	DLC > 3.0  ( for pytorch support), check dlc github for instructions https://github.com/DeepLabCut/DeepLabCut, make sure to get pytorch with Cuda to use the GPU for training
	cv2

Instructions:

(0. copy to local drive)
1. dlc -> open config.yaml
	-change Project path to the folder containg the config.yaml

2.code -> open analyse_pupil
	- code has three sections
		- first, loads needed packages ( wait until fully loaded
		- second : User Variables
			VIDEO_FOLDER = path to a folder containing all videos that you want to analyse 
			MIN_CERTAINTY = min_certainty of the dlc model to still consider it. Important for Downstream analysis (e.g. Interpolating diameter from pupil width instead of height, or detecting blinks
			MAKE_LABELED_VIDEO = if TRUE creates cropped viedeos with tracked points overlayed
			CONFIG_PATH = path to config.yaml
		- third: calls functions, just run whole block
				- Opens a still frame from the middle of each video
				- draw a rectangle by clicking and dragging the mouse, confirm with Enter (instructions are also printed in the console)
				- repeat for each video

What is it doing 
 - Tracks 9 Points ( 4pupil, 4 eye boarder, 1 nose (is optional) )
 - calculates Diameter as the distance between top and bottom pupil
 - infers diameter from from left and right pupil if certainty of top and bottom is to low
 - detects blinks ( top and bottom eye are close together and the pupil points have low certainty) and blanks (replces with NA) points during these times

Input = .Mp4 video ( can be changed by editing the first line in the third code block (  videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")) ) 

Outputs: 1 Folder per Video:
		- original video
		- trimmed video with tracking points
		- some DLC internal .pickle files
		- one H5 and one CSV containing the positiond and lieklihood of each point at each frame
		- blinks.npy = framenumbers where blinks occured
		- diameter.npy = eye diameter per frame
		- xyPos.npy = position of eye center per frame ( direction the eye was looking)	
		- xyPos_diameter_blinks.png: summary graphic (grey bars are blinks)


Retraining:
 refer to DLC github: https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html
