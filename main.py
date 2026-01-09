import time
from pathlib import Path
import csv
import os
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import zmq
import tkinter as tk
from tkinter import simpledialog




################ Constants ###########
WINDOW_NAME = 'Pointing Monitor (Press Q to exit)'
ROI_RADIUS = 40 # Radius in pixels for centroid fit
RESOLUTION = 16 # Resolution of the grid on the ablation target
######################################


############ Where to log the MOT Counts and Shot Log ###########
logFile = ".npy"

##################################################################
#################### Settings window ############################
def open_settings_window():
    def save_settings():
        nonlocal settings
        # Collect values from the entry boxes
        settings = [entry.get() for entry in entries]
        print("Settings saved!")
        print(settings)
        root.quit()  # Close the settings window

    settings = []
    root = tk.Tk()
    root.title("Settings")

    # Example settings labels
    settings_labels = ["Low Threshold", "# of Shots Below Threshold Allowed", "MOT Log Save Filepath", "Shot Log Save Filepath"]
    defaults = [threshold, warning_number, MOTLogFile, ShotLogFile]
    # Create a list to hold entry widgets
    entries = []

    for i in range(0,len(settings_labels)):
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=5)

        label = tk.Label(frame, text=settings_labels[i])
        label.pack(side=tk.LEFT)

        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        entries.append(entry)
        entry.insert(0, defaults[i])

    # Add a Save button
    save_button = tk.Button(root, text="Save", command=save_settings)
    save_button.pack(pady=10)

    root.mainloop()
    root.destroy()
    return settings


#####################################################

####################################################
### ZMQ Socket for recieving MOT Count data
port = '55555'
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:48766")
socket.setsockopt_string(zmq.SUBSCRIBE, "")
####################################################



def find_laser_centroids(frame, n_lasers, threshold=100, display_results=True):
    """

    Finds the centroids of n Gaussian laser beams on a black background image.


    Args:

        image_path: Path to the image file.

        n_lasers: The expected number of laser beams.

        threshold: Threshold value for image binarization (adjust as needed).

        display_results: Whether to display the image with detected centroids.


    Returns:

        A NumPy array of shape (n_lasers, 2) containing the (x, y) coordinates

        of the laser centroids, or None if fewer than n_lasers are detected.

    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    if len(contours) < n_lasers:

        return None # Return None if not enough lasers are detected


    centroids = []

    for contour in contours:

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cX = int(M["m10"] / M["m00"])

            cY = int(M["m01"] / M["m00"])

            centroids.append((cX, cY))


    if len(centroids) < n_lasers:  # Double-check in case some moments were zero

        return None


    centroids = np.array(centroids[:n_lasers])

    if display_results:

        for cX, cY in centroids:

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    return centroids, frame


#########################################
########### Start the video capture
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FOCUS, 0)
vid.set(cv2.CAP_PROP_EXPOSURE, -5)
############################################


####################### Function to draw where target is
def draw_cross(img, x, y, color):
    cv2.line(img, (x-20, y), (x+20, y), color, 3)
    cv2.line(img, (x, y-20), (x, y+20), color, 3)



# Smoothing
spot_x_queue = []
spot_y_queue = []



########### Begin main loop

while True:
    
    ret, frame = vid.read() #Get a new video frame from the camera
    frame = np.array(frame, dtype=np.uint8)

    # Process image
    cropped = frame[140:340, 160:350].copy() # Crop to ROI
    h, w, _ = cropped.shape
    cropped = cv2.resize(cropped, (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
    red = cropped[:,:,2] # Extract red channel

    # Extract location
    x_marginals = red.sum(axis=0)
    y_marginals = red.sum(axis=1)

    spot_x = np.argmax(x_marginals)
    spot_y = np.argmax(y_marginals)

    # Check total intensity
    intensity = red.sum()
    valid = intensity > 200e3

    if valid:
        # Compute correction from centroid fit
        d = np.arange(-ROI_RADIUS, ROI_RADIUS)
        dx = np.average(d, weights=x_marginals[spot_x-ROI_RADIUS : spot_x+ROI_RADIUS])
        dy = np.average(d, weights=y_marginals[spot_y-ROI_RADIUS : spot_y+ROI_RADIUS])
        spot_x += dx
        spot_y += dy

        # Smoothing via moving average
        spot_x_queue.append(spot_x)
        spot_y_queue.append(spot_y)
        spot_x_queue = spot_x_queue[-5:]
        spot_y_queue = spot_y_queue[-5:]
        spot_x = np.mean(spot_x_queue)
        spot_y = np.mean(spot_y_queue)

        # Visually indicate location
        draw_cross(cropped, round(spot_x), round(spot_y), (0, 0, 255))
        cv2.circle(cropped, (round(spot_x), round(spot_y)), ROI_RADIUS, (0, 0, 255), 2)

    # Convert to target coords
    #target_coords = np.linalg.solve(mat, np.array([spot_x, spot_y]) - offset)

    # Display information
    h, w, _ = cropped.shape
    target_h = h
    info_display = np.zeros((h, h, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX


    # Visually indicate location
    if valid:
        target_x = round(spot_x)
        target_y = round(spot_y)
        draw_cross(info_display, target_x, target_y, (0, 255, 0))
    else:
        target_x, target_y = None, None


# Check the socket for a new message, indicating a new shot
    try:
        message = socket.recv_string(flags=zmq.NOBLOCK)
        newShot = True
    except zmq.Again:
        # No message was available to be received
        #print("No message received.")
        newShot = False
    except:
        continue



    display = np.hstack([info_display, cropped]) #No heat map, don't show the heat map
   

    dh, dw, _ = display.shape
    cv2.imshow(WINDOW_NAME, display)

##############################################
#Handle User key input
    key = cv2.waitKey(1) & 0xFF

    #If the user presses q, save the MOT Grid to a file and gracefully exit
    if key == ord('q'):
        np.save('old/frame.npy', frame)
        break

    #If the user presses l, load a raster path from a file
   

    #If you push s, open the settings window
    elif key ==ord('s'):
        returned_settings = open_settings_window()
        threshold = int(returned_settings[0])
        warning_number = int(returned_settings[1])
        MOTLogFile = str(returned_settings[2])
        ShotLogFile = str(returned_settings[3])


    


    #If you push escape, disable any destination locking, autoMove, rastering, etc.
    elif key == 27: # Esc
        dest_x = None
        dest_y = None
        csv_x = None
        csv_y = None
        csv_dur = None
        autoMode = False
        warnMode = False

    

    #end = time.time() #timing for benchmarking
    #print(start-end)

#Gracefully terminate the program upopn breaking from the running loop
vid.release()
cv2.destroyAllWindows()

