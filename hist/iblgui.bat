@echo on
REM Replace hist with your actual conda environment name
call C:\Users\Freitag\AppData\Local\anaconda3\Scripts\activate.bat hist

REM Run your Python script with the argument
python "C:\Users\Freitag\iblapps\atlaselectrophysiology\ephys_atlas_gui.py" -o True

pause
