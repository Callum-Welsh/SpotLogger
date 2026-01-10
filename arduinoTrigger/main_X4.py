import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import serial.tools.list_ports
import numpy as np
import cv2
# from scipy.optimize import curve_fit # Not used currently
# from scipy.signal import find_peaks # Not used currently
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from PIL import Image, ImageTk # Not used directly anymore
import threading
import time
import csv
import os
# from collections import deque # Using lists + slicing for now
import pylablib as pll
from pylablib.devices import Thorlabs
import matplotlib.gridspec as gridspec  # For potentially better layout later

# --- Configuration ---
MAX_CENTROIDS_DISPLAY = 100  # How many *past* centroids to *plot* per beam (reduces plot lag)


def make_avg(avg_dict, running_dict):
    """Calculates average centroid positions for each beam."""
    # (Function unchanged)
    if not running_dict:
        return avg_dict

    new_avg_dict = avg_dict.copy()
    for beam_id, history in running_dict.items():
        if not history:
            continue

        sum_x = sum(p[0] for p in history)
        sum_y = sum(p[1] for p in history)
        count = len(history)

        avg_x = sum_x / count
        avg_y = sum_y / count

        if beam_id in new_avg_dict:
            if isinstance(new_avg_dict[beam_id], list):
                new_avg_dict[beam_id].append([avg_x, avg_y])
            else:
                new_avg_dict[beam_id] = [[avg_x, avg_y]]
        else:
            new_avg_dict[beam_id] = [[avg_x, avg_y]]

    return new_avg_dict


# --- Main Application Class ---
class BeamProfilerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Beam Profiler")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Settings Variables ---
        self.settings = {
            'arduino_port': tk.StringVar(value="COM3"),
            'baud_rate': tk.IntVar(value=115200),
            'camera_index': tk.IntVar(value=0),
            'exposure_time_ms': tk.DoubleVar(value=15),
            'num_avg': tk.IntVar(value=50),
            'save_path': tk.StringVar(value="centroids.csv"),
            'threshold_value': tk.IntVar(value=100),
            'min_blob_area': tk.IntVar(value=300),
        }

        # --- Application State ---
        self.is_listening = False
        self.is_saving = False
        self.arduino_ser = None
        self.camera = None
        self.listen_thread = None
        self.capture_thread = None

        # Data storage
        self.centroids_history = {}
        self.mean_centroids = {}
        self.current_centroids = []
        self.last_image = None
        self.current_contours = []

        self.beam_colors = ["r", "g", "b", "c", "m", "y", "k"]

        self.update_lock = threading.Lock()

        # --- GUI Elements (initialized in setup_gui) ---
        self.fig_plot = None
        self.plot_canvas = None
        self.beam_axes = {}  # Dictionary: beam_id -> matplotlib Axes object
        self.plot_lines = {}  # Dictionary: beam_id -> [hist_line, mean_line]
        self.fig_image = None
        self.ax_image = None
        self.image_plot = None
        self.centroid_markers_plot = []
        self.contour_plots = []
        self.status_label = None
        self.start_stop_button = None
        self.start_stop_save_button = None

        # --- GUI Setup ---
        self.setup_gui()
        self.update_status("Application Started. Connect Hardware.")
        self.batch_counter = 0
        self.num_avg = self.settings['num_avg'].get()

        # self.find_arduino_port()

    def setup_gui(self):
        # Clear previous widgets if necessary
        for widget in self.root.winfo_children():
            widget.destroy()

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Control Frame ---
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.start_stop_button = ttk.Button(control_frame, text="Start Listening", command=self.toggle_listening)
        self.start_stop_save_button = ttk.Button(control_frame, text="Start Saving", command=self.toggle_saving,
                                                 state=tk.DISABLED)
        settings_button = ttk.Button(control_frame, text="Settings", command=self.open_settings)
        clear_button = ttk.Button(control_frame, text="Clear Graph", command=self.clear_graph_data)

        self.start_stop_button.grid(row=0, column=0, padx=5, pady=5)
        self.start_stop_save_button.grid(row=0, column=1, padx=5, pady=5)
        settings_button.grid(row=0, column=2, padx=5, pady=5)
        clear_button.grid(row=0, column=3, padx=5, pady=5)
        self._update_button_states()  # Set initial button states

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
        # Use simpler imshow setup if cv2.applyColorMap handles the colormap
        self.image_plot = self.ax_image.imshow(np.zeros((100, 100, 3), dtype=np.uint8), aspect='auto')
        self.centroid_markers_plot = []
        self.contour_plots = []  # Using cv2.drawContours, so this might not be needed

        self.image_canvas = FigureCanvasTkAgg(self.fig_image, master=image_frame)
        self.image_widget = self.image_canvas.get_tk_widget()
        self.image_widget.pack(fill=tk.BOTH, expand=True)

        # --- Plot Display ---
        plot_frame = ttk.LabelFrame(display_frame, text="Centroid History (Per Beam)", padding="5")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        display_frame.columnconfigure(1, weight=1)

        # Create figure for plots, but add axes dynamically
        self.fig_plot = plt.Figure(figsize=(5, 4))  # Use Figure object directly
        # self.fig_plot.subplots_adjust(hspace=0.4) # Add some vertical space between plots

        self.plot_canvas = FigureCanvasTkAgg(self.fig_plot, master=plot_frame)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.pack(fill=tk.BOTH, expand=True)

        # Reset data stores associated with plots
        self.clear_graph_data(update_display=False)
        self.update_displays()  # Draw initial state (empty plots)

    def _update_button_states(self):
        """Helper to set button states based on app state."""
        if not self.start_stop_button: return  # Avoid errors if called before setup

        if self.is_listening:
            self.start_stop_button.config(text="Stop Listening")
            self.start_stop_save_button.config(state=tk.NORMAL)
            if self.is_saving:
                self.start_stop_save_button.config(text="Stop Saving")
            else:
                self.start_stop_save_button.config(text="Start Saving")
        else:
            self.start_stop_button.config(text="Start Listening")
            self.start_stop_save_button.config(text="Start Saving", state=tk.DISABLED)

    def clear_graph_data(self, update_display=True):
        """Clears the data used for plotting and resets relevant states."""
        with self.update_lock:
            self.centroids_history.clear()
            self.mean_centroids.clear()
            self.current_centroids = []
            self.current_contours = []
            self.batch_counter = 0
            self.last_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Reset image display placeholder

        # Clear plot lines data structure
        self.plot_lines.clear()

        # Remove all axes from the plot figure
        if self.fig_plot:
            # Iterate safely while removing
            for ax in list(self.fig_plot.axes):
                ax.remove()  # Use remove() method of the Axes object
        self.beam_axes.clear()  # Clear the axes dictionary

        if update_display:
            # Redraw the empty figure and update image
            if self.plot_canvas: self.plot_canvas.draw_idle()
            self.update_displays(no_centroids=True)
            self.update_status("Graph data cleared.")

    def update_status(self, message):
        # (Function unchanged)
        def _update():
            if self.status_label:  # Check if label exists
                self.status_label.config(text=f"Status: {message}")

        if hasattr(self.root, 'after'):
            self.root.after(0, _update)
        else:
            _update()
        print(message)

    # --- Hardware Connect/Disconnect, Listener, Trigger Handling ---
    # (These functions remain largely unchanged from the previous threaded version)
    def find_arduino_port(self):
        # (Function unchanged)
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'arduino' in port.description.lower() or 'usb serial' in port.description.lower() or 'ch340' in port.description.lower():
                self.settings['arduino_port'].set(port.device)
                self.update_status(f"Auto-detected Arduino on {port.device}")
                return
        self.update_status("Arduino not auto-detected. Check Settings.")

    def toggle_listening(self):
        # (Function unchanged)
        if not self.is_listening:
            if self.connect_hardware():
                self.is_listening = True
                self.update_status("Listening for trigger...")
                self.listen_thread = threading.Thread(target=self.listen_for_trigger, daemon=True)
                self.listen_thread.start()
            else:
                self.update_status("Failed to connect hardware. Check Settings.")
        else:
            self.is_listening = False
            self.disconnect_hardware()
            self.toggle_saving(force_stop=True)
            self.update_status("Stopped listening.")
        self._update_button_states()  # Update buttons after state change

    def connect_hardware(self):
        # (Function unchanged)
        port = self.settings['arduino_port'].get()
        baud = self.settings['baud_rate'].get()
        if not port:
            messagebox.showerror("Connection Error", "Arduino port not set in Settings.")
            return False
        try:
            self.arduino_ser = serial.Serial(port, baud, timeout=1, write_timeout=1)
            time.sleep(2.0)
            self.arduino_ser.reset_input_buffer()
            self.arduino_ser.reset_output_buffer()
            self.update_status(f"Connected to Arduino on {port}")
        except serial.SerialException as e:
            messagebox.showerror("Arduino Connection Error", f"Failed to connect to {port}: {e}")
            self.arduino_ser = None
            return False
        except Exception as e:
            messagebox.showerror("Arduino Connection Error", f"An unexpected error occurred with {port}: {e}")
            self.arduino_ser = None
            return False

        try:
            cam_list = Thorlabs.list_cameras_tlcam()
            print(f"Detected Thorlabs cameras: {cam_list}")
            if not cam_list:
                raise Exception("No Thorlabs cameras detected by pylablib.")
            self.camera = Thorlabs.ThorlabsTLCamera(cam_list[0])
            exp_time_sec = self.settings['exposure_time_ms'].get() / 1000.0
            self.camera.set_exposure(exp_time_sec)
            self.update_status(f"Connected to Camera! Exposure: {exp_time_sec * 1000:.2f} ms")
        except Exception as e:
            print(f"Camera Connection Error: {e}")
            messagebox.showerror("Camera Connection Error", f"Failed to connect or configure camera: {e}")
            self.camera = None
            if self.arduino_ser and self.arduino_ser.is_open:
                self.arduino_ser.close()
                self.arduino_ser = None
            return False
        return True

    def disconnect_hardware(self):
        # (Function unchanged)
        if self.arduino_ser and self.arduino_ser.is_open:
            try:
                self.arduino_ser.close()
            except Exception as e:
                print(f"Error closing Arduino port: {e}")
            self.arduino_ser = None
            self.update_status("Disconnected Arduino.")
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")
            self.camera = None
            self.update_status("Disconnected Camera.")

    def listen_for_trigger(self):
        # (Function unchanged)
        while self.is_listening:
            if self.arduino_ser and self.arduino_ser.is_open:
                try:
                    if not self.is_listening: break
                    if self.arduino_ser.in_waiting > 0:
                        line = self.arduino_ser.readline().decode('utf-8', errors='ignore').strip()
                        if line == "!":
                            print("Trigger received!")
                            if self.capture_thread is None or not self.capture_thread.is_alive():
                                self.capture_thread = threading.Thread(target=self._capture_and_analyze_task,
                                                                       daemon=True)
                                self.capture_thread.start()
                            else:
                                print("Skipping trigger, previous capture/analysis still running.")

                except serial.SerialException as e:
                    print(f"Serial error during listening: {e}")
                    self.root.after(0, self.handle_serial_error)
                    break
                except UnicodeDecodeError:
                    print("Serial decode error, ignoring.")
                    pass
                except Exception as e:
                    print(f"Unexpected error in listener thread: {e}")
                    time.sleep(0.1)
            time.sleep(0.005)
        print("Listener thread finished.")

    def handle_serial_error(self):
        # (Function unchanged)
        if self.is_listening:
            messagebox.showerror("Serial Error", "Lost connection to Arduino or read error.")
            self.toggle_listening()

    def _capture_and_analyze_task(self):
        # (Function unchanged)
        if not self.is_listening: return
        capture_success, image_data = self.capture_image()
        analysis_success = False
        centroids = []
        contours = []
        if capture_success and image_data is not None:
            centroids, contours = self.analyze_image(image_data)
            analysis_success = True
        else:
            print("Capture failed, skipping analysis.")
        self.root.after(0, self._process_capture_results, capture_success, analysis_success, image_data, centroids,
                        contours)

    def _process_capture_results(self, capture_success, analysis_success, image_data, centroids, contours):
        # (Function unchanged - calls update_displays)
        if not self.is_listening:
            print("Ignoring results received after stopping listener.")
            return

        if not capture_success:
            self.update_status("Image capture failed.")
            return

        self.last_image = image_data
        self.current_centroids = centroids if centroids is not None else []
        self.current_contours = contours if contours is not None else []

        if not analysis_success:
            self.update_status("Image analysis failed (capture OK).")
            self.update_displays(no_centroids=True)
            return

        num_found = len(self.current_centroids)
        if num_found > 0:
            self.batch_counter += 1
            self.update_status(f"Analysis complete. Found {num_found} beams. Batch {self.batch_counter}/{self.num_avg}")

            with self.update_lock:
                for beam_id in range(num_found):
                    if beam_id not in self.centroids_history:
                        self.centroids_history[beam_id] = []
                    self.centroids_history[beam_id].append(self.current_centroids[beam_id])

            if self.batch_counter >= self.num_avg:
                self.update_status(f"Batch complete ({self.num_avg} images). Calculating average...")
                with self.update_lock:
                    self.mean_centroids = make_avg(self.mean_centroids, self.centroids_history)
                    self.centroids_history.clear()
                self.batch_counter = 0
                self.update_status(f"Averaging complete. Found {num_found} beams.")

            self.update_displays()

            if self.is_saving:
                self.save_centroids(self.current_centroids)

        else:
            self.update_status("Analysis complete. No beams found.")
            self.update_displays(no_centroids=True)

    def capture_image(self):
        # (Function unchanged)
        if not self.camera:
            print("Capture Error: Camera not connected.")
            return False, None
        try:
            frame = self.camera.snap()
            if frame is None or frame.size == 0:
                print("Capture Error: Camera returned invalid frame.")
                return False, None
            return True, frame
        except Exception as e:
            print(f"Error during camera snap or processing: {e}")
            return False, None

    def analyze_image(self, image):
        # (Function unchanged)
        if image is None or image.size == 0: return [], []
        try:
            threshold_val = self.settings['threshold_value'].get()
            min_area = self.settings['min_blob_area'].get()
        except tk.TclError as e:
            print(f"Warning: Could not get settings for analysis: {e}")
            threshold_val = 100
            min_area = 100

        if image.dtype != np.uint8:
            img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            img_8bit = image

        _, thresh_img = cv2.threshold(img_8bit, threshold_val, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        valid_contours = []
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area: continue
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centroids.append([np.round(cx, 2), np.round(cy, 2)])
                    valid_contours.append(cnt)
        centroids.sort(key=lambda p: p[0])
        return centroids, valid_contours

    # --- Display Update ---
    def update_displays(self, no_centroids=False):
        """Updates the image display and the centroid plot (now with multiple axes)."""

        # --- Update Image Display ---
        # (Using cv2.applyColorMap approach from previous step)
        if self.last_image is not None and self.last_image.size > 0:
            if len(self.last_image.shape) == 3:
                gray_image = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.last_image.copy()

            if gray_image.dtype != np.uint8:
                gray_image_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                gray_image_norm = gray_image

            display_img_color = cv2.applyColorMap(gray_image_norm, cv2.COLORMAP_JET)

            if self.current_contours:
                cv2.drawContours(display_img_color, self.current_contours, -1, (0, 0, 0), 10)

            self.image_plot.set_data(display_img_color)
            h, w = display_img_color.shape[:2]
            self.image_plot.set_extent((0, w, h, 0))
            self.ax_image.set_xlim(0, w)
            self.ax_image.set_ylim(h, 0)

            for marker in self.centroid_markers_plot: marker.remove()
            self.centroid_markers_plot.clear()

            if not no_centroids and self.current_centroids:
                xs = [c[0] for c in self.current_centroids]
                ys = [c[1] for c in self.current_centroids]
                new_markers = self.ax_image.plot(xs, ys, '+', color='k', markersize=8, markeredgewidth=1.5)
                self.centroid_markers_plot.extend(new_markers)

            self.image_canvas.draw_idle()

        # --- Update Centroid Plot (Multiple Axes) ---
        needs_plot_redraw = False
        rebuilt_layout = False

        with self.update_lock:
            # Determine all beam IDs that have data (current, history, or mean)
            active_beam_ids = set(range(len(self.current_centroids) if not no_centroids else 0)) | \
                              set(self.centroids_history.keys()) | \
                              set(self.mean_centroids.keys())

            if not active_beam_ids:  # No beams to plot
                if self.fig_plot and self.fig_plot.axes:  # Clear existing axes if no beams are active
                    print("No active beams, clearing plot axes.")
                    for ax in list(self.fig_plot.axes):
                        ax.remove()
                    self.beam_axes.clear()
                    self.plot_lines.clear()
                    needs_plot_redraw = True
                # else: pass # Already empty

            else:  # Beams are active, manage axes
                max_needed_id = max(active_beam_ids) if active_beam_ids else -1
                required_rows = max_needed_id + 1
                current_rows = len(self.fig_plot.axes)  # Check how many axes are currently in the figure

                # --- Rebuild Layout if Necessary ---
                if required_rows != current_rows:
                    print(f"Rebuilding plot layout for {required_rows} beams (was {current_rows}).")
                    rebuilt_layout = True
                    needs_plot_redraw = True

                    # Store existing line data before clearing
                    old_lines_data = {}
                    for bid, lines in self.plot_lines.items():
                        old_lines_data[bid] = (lines[0].get_data(), lines[1].get_data())

                    # Clear old axes and dictionaries
                    for ax in list(self.fig_plot.axes):
                        ax.remove()
                    self.beam_axes.clear()
                    self.plot_lines.clear()

                    # Create new axes layout
                    if required_rows > 0:
                        # gs = gridspec.GridSpec(required_rows, 1, figure=self.fig_plot) # Alternative layout
                        for i in range(required_rows):
                            # ax = self.fig_plot.add_subplot(gs[i, 0]) # Using GridSpec
                            ax = self.fig_plot.add_subplot(required_rows, 1, i + 1)  # Simple vertical stack
                            ax.set_xlabel("X Position (px)", fontsize='small')
                            ax.set_ylabel("Y Position (px)", fontsize='small')
                            ax.tick_params(axis='both', which='major', labelsize='small')
                            ax.grid(True)
                            self.beam_axes[i] = ax  # Store by index (beam_id)

                            # Recreate lines if old data exists for this beam_id
                            if i in old_lines_data:
                                hist_data, mean_data = old_lines_data[i]
                                color = self.beam_colors[i % len(self.beam_colors)]
                                line_hist, = ax.plot(hist_data[0], hist_data[1], marker='.', linestyle='-', color=color)
                                line_mean, = ax.plot(mean_data[0], mean_data[1], marker='x', linestyle='None',
                                                     markersize=8, color='k')
                                self.plot_lines[i] = [line_hist, line_mean]
                                ax.set_title(f'Beam {i + 1}', fontsize='medium')  # Set title during recreation
                                ax.relim()  # Recalculate limits after replotting
                                ax.autoscale_view(True, True, True)

                # --- Update Data on Axes ---
                for beam_id in active_beam_ids:
                    if beam_id not in self.beam_axes:
                        # This case should ideally be handled by the rebuild logic above
                        print(f"Warning: Beam ID {beam_id} has data but no axis found after layout check.")
                        continue  # Skip if axis doesn't exist

                    ax = self.beam_axes[beam_id]
                    history_data = self.centroids_history.get(beam_id, [])
                    mean_data = self.mean_centroids.get(beam_id, [])

                    # Get or create plot lines for this axis
                    if beam_id not in self.plot_lines:
                        # Should only happen if layout was rebuilt and old data didn't exist
                        color = self.beam_colors[beam_id % len(self.beam_colors)]
                        line_hist, = ax.plot([], [], marker='.', linestyle='-', color=color)
                        line_mean, = ax.plot([], [], marker='x', linestyle='None', markersize=8, color='k')
                        self.plot_lines[beam_id] = [line_hist, line_mean]
                        ax.set_title(f'Beam {beam_id + 1}', fontsize='medium')  # Set title when creating lines
                        needs_plot_redraw = True  # Need redraw if lines were created

                    line_hist, line_mean = self.plot_lines[beam_id]

                    # Update history line data (limited points)
                    display_history = history_data[-MAX_CENTROIDS_DISPLAY:]
                    hist_x = [p[0] for p in display_history]
                    hist_y = [p[1] for p in display_history]
                    # Only update if data actually changed to avoid unnecessary redraws
                    if not np.array_equal(line_hist.get_xdata(), hist_x) or not np.array_equal(line_hist.get_ydata(),
                                                                                               hist_y):
                        line_hist.set_data(hist_x, hist_y)
                        needs_plot_redraw = True

                    # Update mean line data
                    avg_x = [p[0] for p in mean_data]
                    avg_y = [p[1] for p in mean_data]
                    if not np.array_equal(line_mean.get_xdata(), avg_x) or not np.array_equal(line_mean.get_ydata(),
                                                                                              avg_y):
                        line_mean.set_data(avg_x, avg_y)
                        needs_plot_redraw = True

                    # Rescale axis if data was updated for it
                    if needs_plot_redraw and not rebuilt_layout:  # Avoid rescaling if layout was just rebuilt (already done)
                        ax.relim()
                        ax.autoscale_view(True, True, True)

        # --- Redraw Plot Canvas ---
        if needs_plot_redraw and self.plot_canvas:
            try:
                # Adjust layout to prevent overlap, especially after adding/removing axes
                self.fig_plot.tight_layout()
                # self.fig_plot.subplots_adjust(hspace=0.4) # Manual adjustment if tight_layout isn't enough
            except Exception as e:
                print(f"Error during tight_layout: {e}")  # Can sometimes fail with complex layouts
            self.plot_canvas.draw_idle()

    # --- Saving Logic ---
    # (Functions toggle_saving, save_centroids unchanged)
    def toggle_saving(self, force_stop=False):
        # (Function unchanged)
        if force_stop:
            if self.is_saving:
                self.is_saving = False
                self.update_status("Saving stopped.")
            self._update_button_states()  # Ensure button state correct after force stop
            return

        if not self.is_listening and not self.is_saving:
            messagebox.showwarning("Save Error", "Cannot start saving when not listening.")
            return

        if not self.is_saving:
            fpath = self.settings['save_path'].get()
            if not fpath:
                messagebox.showerror("Save Error", "No save file path specified in Settings.")
                return
            write_header = not os.path.exists(fpath) or os.path.getsize(fpath) == 0
            try:
                with open(fpath, 'a', newline='') as f:
                    if write_header:
                        writer = csv.writer(f)
                        max_expected_beams = 5
                        header = ['Timestamp']
                        for i in range(max_expected_beams):
                            header.extend([f'Beam_{i + 1}_X', f'Beam_{i + 1}_Y'])
                        writer.writerow(header)
                self.is_saving = True
                self.update_status(f"Saving centroids to {fpath}")
            except IOError as e:
                messagebox.showerror("Save Error", f"Cannot open file {fpath} for writing: {e}")
        else:
            self.is_saving = False
            self.update_status("Saving stopped.")
        self._update_button_states()  # Update buttons after state change

    def save_centroids(self, centroids):
        # (Function unchanged)
        if not self.is_saving or not centroids: return
        fpath = self.settings['save_path'].get()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        row = [timestamp]
        for x, y in centroids:
            row.extend([f"{x:.4f}", f"{y:.4f}"])
        try:
            with open(fpath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            self.update_status(f"Error saving data: {e}")

    # --- Settings Window ---
    # (Functions open_settings, browse_save_path, apply_settings unchanged)
    def open_settings(self):
        # (Function unchanged)
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.transient(self.root)
        settings_win.grab_set()
        frame = ttk.Frame(settings_win, padding="15")
        frame.pack(expand=True, fill=tk.BOTH)
        row = 0
        ttk.Label(frame, text="Arduino Port:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['arduino_port'], width=40).grid(row=row, column=1, columnspan=2,
                                                                                    sticky=tk.EW, padx=5)
        row += 1
        ttk.Label(frame, text="Baud Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['baud_rate']).grid(row=row, column=1, columnspan=2, sticky=tk.EW,
                                                                       padx=5)
        row += 1
        ttk.Label(frame, text="Camera Index/ID (Info):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['camera_index']).grid(row=row, column=1, columnspan=2, sticky=tk.EW,
                                                                          padx=5)
        row += 1
        ttk.Label(frame, text="Exposure (ms):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['exposure_time_ms']).grid(row=row, column=1, columnspan=2,
                                                                              sticky=tk.EW, padx=5)
        row += 1
        ttk.Label(frame, text="Images per Avg Batch:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['num_avg']).grid(row=row, column=1, columnspan=2, sticky=tk.EW,
                                                                     padx=5)
        row += 1
        ttk.Label(frame, text="Save File Path:").grid(row=row, column=0, sticky=tk.W, pady=2)
        save_entry = ttk.Entry(frame, textvariable=self.settings['save_path'], width=35)
        save_entry.grid(row=row, column=1, sticky=tk.EW, padx=5)
        browse_button = ttk.Button(frame, text="...", width=3, command=lambda: self.browse_save_path(settings_win))
        browse_button.grid(row=row, column=2, sticky=tk.W)
        row += 1
        ttk.Label(frame, text="Threshold (0-255):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['threshold_value']).grid(row=row, column=1, columnspan=2,
                                                                             sticky=tk.EW, padx=5)
        row += 1
        ttk.Label(frame, text="Min Blob Area (px):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=self.settings['min_blob_area']).grid(row=row, column=1, columnspan=2,
                                                                           sticky=tk.EW, padx=5)
        row += 1
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ok_button = ttk.Button(button_frame, text="Apply & Close", command=lambda: self.apply_settings(settings_win))
        ok_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=settings_win.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)
        frame.columnconfigure(1, weight=1)

    def browse_save_path(self, parent):
        # (Function unchanged)
        fpath = filedialog.asksaveasfilename(
            parent=parent,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self.settings['save_path'].get() or "centroids.csv",
            title="Select Save File"
        )
        if fpath:
            self.settings['save_path'].set(fpath)

    def apply_settings(self, settings_win):
        # (Function unchanged)
        try:
            baud = self.settings['baud_rate'].get()
            exp_ms = self.settings['exposure_time_ms'].get()
            num_avg = self.settings['num_avg'].get()
            thresh = self.settings['threshold_value'].get()
            min_area = self.settings['min_blob_area'].get()
            if num_avg < 1: num_avg = 1; self.settings['num_avg'].set(1)
            if not (0 <= thresh <= 255): thresh = 50; self.settings['threshold_value'].set(50)
            if min_area < 0: min_area = 0; self.settings['min_blob_area'].set(0)
            if exp_ms <= 0: exp_ms = 1; self.settings['exposure_time_ms'].set(1)
            self.num_avg = num_avg  # Update internal state
            if self.camera:
                exp_time_sec = exp_ms / 1000.0
                try:
                    self.camera.set_exposure(exp_time_sec)
                    self.update_status(f"Exposure updated to {exp_ms:.2f} ms")
                except Exception as e:
                    messagebox.showerror("Settings Error", f"Failed to apply exposure setting to camera: {e}",
                                         parent=settings_win)
        except tk.TclError:
            messagebox.showerror("Settings Error", "Invalid numerical value entered.", parent=settings_win)
            return
        except Exception as e:
            messagebox.showerror("Settings Error", f"An error occurred applying settings: {e}", parent=settings_win)
            return
        self.update_status("Settings applied.")
        settings_win.destroy()

    # --- Closing ---
    def on_closing(self):
        # (Function unchanged)
        if self.is_listening:
            if messagebox.askokcancel("Quit", "Listening is active. Stop listening and quit?"):
                self.toggle_listening()
                time.sleep(0.1)  # Brief pause
                self.root.destroy()
            else:
                return
        else:
            self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BeamProfilerApp(root)
    root.mainloop()