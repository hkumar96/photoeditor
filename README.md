#Photoeditor application

This repository consists of the following files:
1. main_app.py: This is the main file to be exectued to run the photoeditor
2. inpainting.py: This file contains inpainting algorithm, this is used in main_app.py
3. mouse.py: This file hosts the code to create a mask from user input.

Main application is GUI photoeditor that provides user with two features:
1. Red eye removal: Detect and remove any red eye if found.
2. Exemplar based inpainting: This implements an exemplar based inpainting algorithm. Refer [this](https://www.computer.org/csdl/proceedings/cvpr/2003/1900/02/190020721-abs.html)
```bash
Usage: $ python main_app.py
```