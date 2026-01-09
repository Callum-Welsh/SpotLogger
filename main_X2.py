import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import serial.tools.list_ports
import cv2 # Placeholder for camera control - REPLACE if using SDK
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import threading
import time
import csv
import os
from collections import deque
import pylablib as pll
from pylablib.devices import Thorlabs

# --- Configuration ---
MAX_CENTROIDS_DISPLAY = 50 # How many past centroids to plot

# --- Gaussian Fitting Function ---
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function"""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

# --- Main Application Class ---
class BeamProfilerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Beam Profiler")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close

        # --- Settings Variables ---
        self.settings = {
            'arduino_port': tk.StringVar(value=""),
            'baud_rate': tk.IntVar(value=9600),
            'camera_index': tk.IntVar(value=0), # For OpenCV, change if using SDK
            'exposure_time_ms': tk.DoubleVar(value=10.0),
            'num_avg': tk.IntVar(value=1),
            'save_path': tk.StringVar(value="centroids.csv"),
            'threshold_value': tk.IntVar(value=50), # Simple threshold
            'min_blob_area': tk.IntVar(value=100), # Min px area for a blob
        }

        # --- Application State ---
        self.is_listening = False
        self.is_saving = False
        self.arduino_ser = None
        self.camera = None # Placeholder for camera object
        self.listen_thread = None
        self.centroids_history = {} # Dictionary: beam_id -> deque of (x, y)
        self.current_centroids = [] # List of (x, y) from last image
        self.last_image = None
        self.beam_colors = plt.cm.viridis(np.linspace(0, 1, 10)) # Colors for plotting different beams

        # --- GUI Setup ---
        self.setup_gui()
        self.update_status("Application Started. Connect Hardware.")

        # --- Auto-detect Arduino Port ---
        self.find_arduino_port()


    def setup_gui(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Control Frame ---
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.start_stop_button = ttk.Button(control_frame, text="Start Listening", command=self.toggle_listening)
        self.start_stop_button.grid(row=0, column=0, padx=5, pady=5)

        self.start_stop_save_button = ttk.Button(control_frame, text="Start Saving", command=self.toggle_saving, state=tk.DISABLED)
        self.start_stop_save_button.grid(row=0, column=1, padx=5, pady=5)

        settings_button = ttk.Button(control_frame, text="Settings", command=self.open_settings)
        settings_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Status Bar ---
        self.status_label = ttk.Label(main_frame, text="Status: Idle")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # --- Display Frame ---
        display_frame = ttk.Frame(main_frame, padding="10")
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # --- Image Display ---
        image_frame = ttk.LabelFrame(display_frame, text="Last Captured Image", padding="5")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        self.fig_image, self.ax_image = plt.subplots(figsize=(5, 4))
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])

        self.ax_image.set_aspect('auto')

        self.image_canvas = FigureCanvasTkAgg(self.fig_image, master=image_frame)
        self.image_widget = self.image_canvas.get_tk_widget()
        self.image_widget.pack(fill=tk.BOTH, expand=True)
        self.image_plot = self.ax_image.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray', vmin=0, vmax=255) # Placeholder
        self.centroid_markers = [] # Store scatter plot objects

        # --- Plot Display ---
        plot_frame = ttk.LabelFrame(display_frame, text="Centroid History", padding="5")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        display_frame.columnconfigure(1, weight=1)

        self.fig_plot, self.ax_plot = plt.subplots(figsize=(5, 4))
        self.ax_plot.set_xlabel("Image Index (Relative)")
        self.ax_plot.set_ylabel("Position (Pixels)")
        self.ax_plot.grid(True)
        self.plot_canvas = FigureCanvasTkAgg(self.fig_plot, master=plot_frame)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.pack(fill=tk.BOTH, expand=True)
        self.plot_lines = {} # Dictionary: beam_id -> matplotlib line object


    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        print(message) # Also print to console

    def find_arduino_port(self):
        """Attempts to find a connected Arduino."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Look for common Arduino identifiers (may need adjustment)
            if 'arduino' in port.description.lower() or 'usb serial' in port.description.lower() or 'ch340' in port.description.lower():
                 self.settings['arduino_port'].set(port.device)
                 self.update_status(f"Auto-detected Arduino on {port.device}")
                 return
        self.update_status("Arduino not auto-detected. Check Settings.")


    # --- Core Logic ---
    def toggle_listening(self):
        if not self.is_listening:
            if self.connect_hardware():
                self.is_listening = True
                self.start_stop_button.config(text="Stop Listening")
                self.start_stop_save_button.config(state=tk.NORMAL) # Enable saving button
                self.update_status("Listening for trigger...")
                # Start listening thread
                self.listen_thread = threading.Thread(target=self.listen_for_trigger, daemon=True)
                self.listen_thread.start()
            else:
                self.update_status("Failed to connect hardware. Check Settings.")
        else:
            self.is_listening = False # Signal thread to stop
            # Wait briefly for thread to notice the flag (better: use threading.Event)
            time.sleep(0.2)
            if self.listen_thread and self.listen_thread.is_alive():
                # Force stop might be needed if thread blocks, but clean exit is better
                pass # Rely on thread checking self.is_listening
            self.disconnect_hardware()
            self.start_stop_button.config(text="Start Listening")
            self.toggle_saving(force_stop=True) # Ensure saving stops if listening stops
            self.start_stop_save_button.config(state=tk.DISABLED)
            self.update_status("Stopped listening.")


    def connect_hardware(self):
        """Connects to Arduino and Camera based on settings."""
        # Connect Arduino
        port = self.settings['arduino_port'].get()
        baud = self.settings['baud_rate'].get()
        if not port:
            messagebox.showerror("Connection Error", "Arduino port not set in Settings.")
            return False
        try:
            self.arduino_ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2) # Allow time for Arduino reset
            self.update_status(f"Connected to Arduino on {port}")
        except serial.SerialException as e:
            messagebox.showerror("Arduino Connection Error", f"Failed to connect to {port}: {e}")
            self.arduino_ser = None
            return False

        # Connect Camera (Placeholder using OpenCV)
        # **** REPLACE THIS SECTION WITH YOUR CAMERA SDK CODE ****
        try:
            self.camera = Thorlabs.ThorlabsTLCamera()
            self.camera.open()
            # --- Try setting exposure (OpenCV method, often limited) ---
            # Note: Exposure units for OpenCV are often logarithmic or camera-specific
            # You'll likely need SDK functions for precise millisecond control
            # cv_exp_val = self.ms_to_cv_exposure(self.settings['exposure_time_ms'].get())
            self.camera.set_exposure(self.settings['exposure_time_ms'].get())
            self.update_status(f"Connected to Camera!")
            # **** END OF PLACEHOLDER SECTION ****

        except Exception as e:
            messagebox.showerror("Camera Connection Error", f"Failed to connect to camera: {e}")
            self.camera = None
            # Disconnect Arduino if camera fails
            if self.arduino_ser and self.arduino_ser.is_open:
                self.arduino_ser.close()
                self.arduino_ser = None
            return False

        return True

    def disconnect_hardware(self):
        """Disconnects from Arduino and Camera."""
        if self.arduino_ser and self.arduino_ser.is_open:
            self.arduino_ser.close()
            self.arduino_ser = None
            self.update_status("Disconnected Arduino.")
        # Disconnect Camera (Placeholder)
        # **** REPLACE WITH YOUR CAMERA SDK DISCONNECT CODE ****
        if self.camera:
            self.camera.release()
            self.camera = None
            self.update_status("Disconnected Camera.")
        # **** END OF PLACEHOLDER ****

    def listen_for_trigger(self):
        """Runs in a separate thread, listens for serial message from Arduino."""
        while self.is_listening:
            if self.arduino_ser and self.arduino_ser.is_open:
                try:
                    if self.arduino_ser.in_waiting > 0:
                        line = self.arduino_ser.readline().decode('utf-8').strip()
                        if line == "TRIGGER":
                            print("Trigger received!")
                            # Schedule image capture and analysis in the main thread
                            self.root.after(0, self.handle_trigger)
                except serial.SerialException:
                    # Handle disconnection mid-listen (optional)
                    self.root.after(0, self.handle_serial_error)
                    break # Exit thread on error
                except UnicodeDecodeError:
                    pass # Ignore occasional garbage data
            time.sleep(0.01) # Small sleep to prevent busy-waiting


    def handle_serial_error(self):
        """Called from listen_thread via root.after if serial fails"""
        messagebox.showerror("Serial Error", "Lost connection to Arduino.")
        self.toggle_listening() # Stop the process


    def handle_trigger(self):
        """Called by listen_thread via root.after when trigger is received."""
        if not self.is_listening: return # Check if we stopped while waiting for main thread

        self.update_status("Trigger received, capturing image...")
        success, image_data = self.capture_image()

        if success:
            self.last_image = image_data # Store the raw captured image
            self.update_status("Image captured, analyzing...")
            centroids = self.analyze_image(image_data)
            self.current_centroids = centroids # Store latest centroids

            if centroids:
                self.update_status(f"Analysis complete. Found {len(centroids)} beams.")
                self.update_displays()
                if self.is_saving:
                    self.save_centroids(centroids)
            else:
                 self.update_status("Analysis complete. No beams found.")
                 # Update display with image but no markers/plot points
                 self.update_displays(no_centroids=True)
        else:
            self.update_status("Image capture failed.")


    def capture_image(self):
        """Captures an image from the camera, potentially averaging."""
        # **** REPLACE WITH YOUR CAMERA SDK CAPTURE CODE ****
        # This section needs modification for:
        # 1. Hardware triggering (if camera supports it directly) instead of software triggering post-Arduino signal.
        # 2. Setting precise exposure time via SDK.
        # 3. Acquiring image data in the correct format (e.g., Bayer, Mono8, Mono12).
        if not self.camera:
            return False, None
        exposure_ms = self.settings['exposure_time_ms'].get()

        # ---- SDK-Specific Exposure Setting would go here ----
        # Example: self.camera.ExposureTime.SetValue(exposure_ms * 1000) # If SDK uses microseconds
        
        frame = self.camera.snap()
            
        # Convert to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame # Assume already grayscale

            # Small delay needed? Depends on camera/trigger setup
            # time.sleep(0.01)

        if gray_frame is None: return False, None
        # Clamp and convert back to uint8 for display/basic processing
        # Note: Keep float version if analysis benefits from higher precision
        avg_image = np.clip(gray_frame, 0, 255).astype(np.uint8)
        return True, avg_image

    def analyze_image(self, image):
        """Analyzes the image to find Gaussian beam centroids."""
        if image is None: return []

        threshold_val = self.settings['threshold_value'].get()
        min_area = self.settings['min_blob_area'].get()

        # 1. Thresholding
        _, thresh_img = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)

        # 2. Find Contours (potential beams)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        height, width = image.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # 3. Get Bounding Box for ROI
            x, y, w, h = cv2.boundingRect(cnt)
            roi = image[y:y+h, x:x+w]
            roi_x_coords = x_coords[y:y+h, x:x+w]
            roi_y_coords = y_coords[y:y+h, x:x+w]

            if roi.size == 0 or np.max(roi) == 0: # Skip empty or black ROIs
                continue

            # 4. Initial Guess for Gaussian Fit
            # Simple guess: Center of bounding box, max intensity, basic width
            M = cv2.moments(cnt) # Can use moments for better initial guess if needed
            if M['m00'] == 0: continue
            cx_guess = M['m10'] / M['m00']
            cy_guess = M['m01'] / M['m00']
            amplitude_guess = np.max(roi)
            offset_guess = np.min(roi) # Or image background level
            sigma_guess = max(w, h) / 4 # Rough guess

            initial_guess = [amplitude_guess, cx_guess, cy_guess, sigma_guess, sigma_guess, 0, offset_guess]
            bounds = ([0, x, y, 0, 0, -np.pi/2, 0], # Lower bounds
                      [2*np.max(image), x+w, y+h, width, height, np.pi/2, np.max(image)]) # Upper bounds

            try:
                # Flatten data for curve_fit
                xy_mesh = np.vstack((roi_x_coords.ravel(), roi_y_coords.ravel()))
                pixel_values = roi.ravel()

                # Filter out zero/low values if they skew the fit
                mask = pixel_values > (offset_guess + (amplitude_guess * 0.1)) # Example: fit points > 10% of peak
                if np.sum(mask) < 10: # Need enough points for fit
                   print(f"Warning: Not enough bright points in ROI {i} for fitting.")
                   # Fallback: Use moments centroid if fit fails?
                   centroids.append((cx_guess, cy_guess)) # Add moments centroid as fallback
                   continue


                popt, pcov = curve_fit(gaussian_2d, xy_mesh[:, mask], pixel_values[mask],
                                       p0=initial_guess, bounds=bounds, maxfev=5000)

                amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
                # Check if fit converged reasonably (e.g., centroid within bounds)
                if x <= x0 < x+w and y <= y0 < y+h:
                     centroids.append((x0, y0))
                else:
                     print(f"Warning: Fit centroid ({x0:.1f}, {y0:.1f}) outside ROI {i}. Using moments centroid.")
                     centroids.append((cx_guess, cy_guess)) # Fallback

            except (RuntimeError, ValueError) as e:
                print(f"Warning: Gaussian fit failed for contour {i}: {e}. Using moments centroid.")
                # Fallback to moments centroid
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centroids.append((cx, cy))

        # Sort centroids (e.g., left-to-right) for consistent plotting ID
        centroids.sort(key=lambda p: p[0])

        return centroids


    def update_displays(self, no_centroids=False):
        """Updates the image display and the centroid plot."""
        # --- Update Image Display ---
        if self.last_image is not None:
            self.image_plot.set_data(self.last_image)
            # Adjust display range if needed (e.g., for > 8-bit images)
            # self.image_plot.set_clim(vmin=np.min(self.last_image), vmax=np.max(self.last_image))

            h, w = self.last_image.shape[:2]
            print(h)
            print(w)
            self.ax_image.set_xlim(0, w)
            self.ax_image.set_ylim(h, 0) # Invert y-axis for image convention

            # Clear previous centroid markers
            for marker in self.centroid_markers:
                marker.remove()
            self.centroid_markers.clear()

            if not no_centroids and self.current_centroids:
                 # Draw new markers
                 xs = [c[0] for c in self.current_centroids]
                 ys = [c[1] for c in self.current_centroids]
                 markers = self.ax_image.plot(xs, ys, 'r+', markersize=10, markeredgewidth=1.5)
                 self.centroid_markers.extend(markers)

            self.image_canvas.draw_idle() # Use draw_idle for efficiency

        # --- Update Centroid Plot ---
        if not no_centroids and self.current_centroids:
            # Assign centroids to history (simple positional matching)
            num_beams = len(self.current_centroids)
            updated_ids = set()

            for beam_id in range(num_beams):
                 if beam_id not in self.centroids_history:
                     self.centroids_history[beam_id] = deque(maxlen=MAX_CENTROIDS_DISPLAY)
                 self.centroids_history[beam_id].append(self.current_centroids[beam_id])
                 updated_ids.add(beam_id)

                 # Get data for this beam
                 history = list(self.centroids_history[beam_id])
                 x_pos = [h[0] for h in history]
                 y_pos = [h[1] for h in history]
                 indices = range(len(history)) # Simple index for x-axis of plot

                 color = self.beam_colors[beam_id % len(self.beam_colors)] # Cycle through colors

                 # Create or update plot lines
                 if beam_id in self.plot_lines:
                      self.plot_lines[beam_id][0].set_data(indices, x_pos) # Update x line
                      self.plot_lines[beam_id][1].set_data(indices, y_pos) # Update y line
                 else:
                      line_x, = self.ax_plot.plot(indices, x_pos, marker='.', linestyle='-', color=color, label=f'Beam {beam_id+1} X')
                      line_y, = self.ax_plot.plot(indices, y_pos, marker='.', linestyle='--', color=color, label=f'Beam {beam_id+1} Y')
                      self.plot_lines[beam_id] = [line_x, line_y]
                      self.ax_plot.legend(fontsize='small') # Update legend if new lines added

            # Remove plot lines for beams that disappeared (optional)
            # ids_to_remove = set(self.plot_lines.keys()) - updated_ids
            # for beam_id in ids_to_remove:
            #      self.plot_lines[beam_id][0].remove()
            #      self.plot_lines[beam_id][1].remove()
            #      del self.plot_lines[beam_id]
            #      del self.centroids_history[beam_id] # Clear history too
            # if ids_to_remove: self.ax_plot.legend(fontsize='small')


            self.ax_plot.relim() # Recalculate axis limits
            self.ax_plot.autoscale_view(True,True,True)
            self.plot_canvas.draw_idle()


    # --- Saving Logic ---
    def toggle_saving(self, force_stop=False):
        if force_stop:
            if self.is_saving:
                 self.is_saving = False
                 self.start_stop_save_button.config(text="Start Saving")
                 self.update_status("Saving stopped.")
            return

        if not self.is_saving:
            # Check if file exists, ask to overwrite or append?
            fpath = self.settings['save_path'].get()
            if not fpath:
                 messagebox.showerror("Save Error", "No save file path specified in Settings.")
                 return

            write_header = not os.path.exists(fpath)
            try:
                 # Test if we can open the file for appending
                 with open(fpath, 'a', newline='') as f:
                     if write_header:
                         writer = csv.writer(f)
                         # Write a flexible header
                         max_expected_beams = 10 # Adjust as needed
                         header = ['Timestamp']
                         for i in range(max_expected_beams):
                             header.extend([f'Beam_{i+1}_X', f'Beam_{i+1}_Y'])
                         writer.writerow(header)

                 self.is_saving = True
                 self.start_stop_save_button.config(text="Stop Saving")
                 self.update_status(f"Saving centroids to {fpath}")
            except IOError as e:
                 messagebox.showerror("Save Error", f"Cannot open file {fpath} for writing: {e}")
        else:
            self.is_saving = False
            self.start_stop_save_button.config(text="Start Saving")
            self.update_status("Saving stopped.")


    def save_centroids(self, centroids):
        """Appends the current centroids to the CSV file."""
        if not self.is_saving or not centroids:
            return

        fpath = self.settings['save_path'].get()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"

        # Create a row with timestamp followed by x,y for each beam
        row = [timestamp]
        for x, y in centroids:
            row.extend([f"{x:.4f}", f"{y:.4f}"])

        try:
            with open(fpath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            self.update_status(f"Error saving data: {e}")
            # Optionally stop saving automatically on error
            # self.toggle_saving()


    # --- Settings Window ---
    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.transient(self.root) # Keep on top of main window
        settings_win.grab_set() # Modal window

        frame = ttk.Frame(settings_win, padding="15")
        frame.pack(expand=True, fill=tk.BOTH)

        row = 0
        # Arduino Port
        ttk.Label(frame, text="Arduino Port:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['arduino_port'], width=40).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Baud Rate
        ttk.Label(frame, text="Baud Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['baud_rate']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Camera Index (OpenCV specific)
        ttk.Label(frame, text="Camera Index/ID:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['camera_index']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Exposure Time
        ttk.Label(frame, text="Exposure (ms):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['exposure_time_ms']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Averaging
        ttk.Label(frame, text="Images to Average:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['num_avg']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Save Path
        ttk.Label(frame, text="Save File Path:").grid(row=row, column=0, sticky=tk.W, pady=2)
        save_entry = ttk.Entry(frame, textvariable=self.settings['save_path'], width=35)
        save_entry.grid(row=row, column=1, sticky=tk.EW, padx=5)
        browse_button = ttk.Button(frame, text="...", width=3, command=lambda: self.browse_save_path())
        browse_button.grid(row=row, column=2, sticky=tk.W)
        row += 1
        # Threshold Value
        ttk.Label(frame, text="Threshold (0-255):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['threshold_value']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1
        # Min Blob Area
        ttk.Label(frame, text="Min Blob Area (px):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['min_blob_area']).grid(row=row, column=1, sticky=tk.EW, padx=5)
        row += 1

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ok_button = ttk.Button(button_frame, text="OK", command=lambda: self.apply_settings(settings_win))
        ok_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=settings_win.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        frame.columnconfigure(1, weight=1) # Make entry widgets expand

    def browse_save_path(self):
        """Opens file dialog to select save path."""
        # Suggest '.csv' extension
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self.settings['save_path'].get() or "centroids.csv",
            title="Select Save File"
        )
        if fpath:
            self.settings['save_path'].set(fpath)

    def apply_settings(self, settings_win):
        """Applies the settings (basic validation can be added here)."""
        # Simple validation example:
        try:
            _ = self.settings['baud_rate'].get()
            _ = self.settings['camera_index'].get()
            _ = self.settings['exposure_time_ms'].get()
            _ = self.settings['num_avg'].get()
            _ = self.settings['threshold_value'].get()
            _ = self.settings['min_blob_area'].get()
            if self.settings['num_avg'].get() < 1: self.settings['num_avg'].set(1)
            if not (0 <= self.settings['threshold_value'].get() <= 255): self.settings['threshold_value'].set(50)
            if self.settings['min_blob_area'].get() < 0: self.settings['min_blob_area'].set(0)
        except tk.TclError:
             messagebox.showerror("Settings Error", "Invalid numerical value entered.", parent=settings_win)
             return

        self.update_status("Settings updated.")
        settings_win.destroy()


    def on_closing(self):
        """Handles window close event."""
        if self.is_listening:
            if messagebox.askokcancel("Quit", "Listening is active. Stop listening and quit?"):
                self.toggle_listening() # Attempt graceful stop
                self.root.destroy()
        else:
            self.root.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BeamProfilerApp(root)
    root.mainloop()