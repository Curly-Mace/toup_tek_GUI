"""

Behaviour:
    Shows a live false-colour video feed in a Tkinter window.
    Provides both slider controls and number entry boxes for camera gain
    and exposure.

Effects:
    Opens the first connected ToupTek camera.
    Starts a camera stream using the ToupTek SDK callback system.
    Displays camera frames in a Tkinter canvas.
    Sends manual gain and exposure changes to the camera.

Returns:
    None directly. This file runs a GUI application when executed.

Params:
    None at file level.

Requires:
    toupcam.py + toupcam.dll in the same folder as this file

Install:
    pip install opencv-python numpy
"""


import sys
import threading

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
import toupcam


# Global camera state
#
# Behaviour:
#     Stores shared camera and frame data used by the SDK callback and GUI.
#
# Effects:
#     These values are read and modified by several functions.
#
# Returns:
#     None.
#
# Params:
#     None.

cam = None
# Camera handle returned by toupcam.Toupcam.Open().
# None means no camera has been opened yet.

frame_buf = None
# Raw byte buffer where the SDK writes 16-bit pixel data.

latest_raw = None
# Most recent camera frame as a NumPy uint16 array.

frame_lock = threading.Lock()
# Protects latest_raw because it is shared between:
#     1. the SDK callback thread
#     2. the Tkinter GUI thread

running = False
# True while the camera is streaming.
# False when the app is closing.

WIDTH = 0
HEIGHT = 0
# Filled after the camera opens.
# These store the camera sensor width and height in pixels.



# Camera defaults


DEFAULT_EXPOSURE_US = 100_000
# Initial exposure time.
# Unit: microseconds.
# 100,000 µs = 100 ms = 0.1 seconds.

DEFAULT_GAIN = 100
# Initial analog gain.
# ToupTek gain units use 100 = 1× gain.
# Examples:
#     100  = 1×
#     200  = 2×
#     1000 = 10×

DEFAULT_TEMP = -10
# Initial TEC cooler target temperature in degrees Celsius.



# Gain limits


GAIN_MIN = 100
# Minimum gain value.
# 100 = 1× gain.

GAIN_MAX = 4800
# Maximum gain value.
# 4800 = 48× gain.


# Exposure limits

EXPO_MIN_US = 100
# Minimum exposure time in microseconds.
# 100 µs = 0.1 ms.

EXPO_MAX_US = 30_000_000
# Maximum exposure time in microseconds.
# 30,000,000 µs = 30 seconds.



# SDK camera event callback

def on_camera_event(event, ctx):
    """
    Behaviour:
        Handles events sent by the ToupTek SDK.

        The only event this function uses is TOUPCAM_EVENT_IMAGE, which means
        the camera has finished an exposure and a new image frame is ready.

        When a frame is ready, the function pulls 16-bit image data from the
        camera into frame_buf, converts that buffer into a NumPy array, and
        stores the result in latest_raw for the GUI to display.

    Effects:
        Reads image data from the camera SDK.
        Writes a new NumPy array into the global latest_raw variable.
        Uses frame_lock so the GUI does not read latest_raw while it is being
        updated.
        Prints an error message if frame capture fails.

    Returns:
        None.

    Params:
        event (int):
            SDK event code. This function only handles
            toupcam.TOUPCAM_EVENT_IMAGE.

        ctx (object):
            Optional user context passed by the SDK.
            This program does not use it, so it is expected to be None.
    """
    global latest_raw

    if event == toupcam.TOUPCAM_EVENT_IMAGE:
        try:
            cam.PullImageV2(frame_buf, 16, None)

            arr = np.frombuffer(
                frame_buf,
                dtype=np.uint16
            ).reshape((HEIGHT, WIDTH)).copy()

            with frame_lock:
                latest_raw = arr

        except Exception as e:
            print(f"[Frame error] {e}")


# Camera open / close helpers

def open_camera():
    """
    Behaviour:
        Searches for connected ToupTek cameras.
        Opens the first detected camera.
        Selects the highest-resolution image size.
        Allocates a frame buffer for 16-bit image data.
        Applies default exposure, gain, and TEC cooler settings.
        Starts pull-mode streaming with on_camera_event as the callback.

    Effects:
        Sets the global variables cam, frame_buf, WIDTH, HEIGHT, and running.
        Sends setup commands to the physical camera.
        Starts the camera stream.

    Returns:
        bool:
            True if a camera was found, opened, configured, and started.
            False if no camera was detected.

    Params:
        None.
    """
    global cam, frame_buf, WIDTH, HEIGHT, running

    devices = toupcam.Toupcam.EnumV2()

    if not devices:
        return False

    cam = toupcam.Toupcam.Open(devices[0].id)

    cam.put_eSize(0)
    WIDTH, HEIGHT = cam.get_Size()

    frame_buf = bytes(WIDTH * HEIGHT * 2)

    cam.put_ExpoTime(DEFAULT_EXPOSURE_US)
    cam.put_ExpoAGain(DEFAULT_GAIN)
    cam.put_AutoExpoEnable(False)

    cam.put_Option(toupcam.TOUPCAM_OPTION_TEC, 1)
    cam.put_Temperature(DEFAULT_TEMP * 10)

    running = True

    cam.StartPullModeWithCallback(on_camera_event, None)

    return True


def close_camera():
    """
    Behaviour:
        Stops the camera stream and closes the camera connection.

        This is called when the user closes the GUI window so the SDK callback
        does not continue running after the Tkinter window has been destroyed.

    Effects:
        Sets running to False.
        Calls cam.Stop() to stop streaming.
        Calls cam.Close() to release the USB camera handle.

    Returns:
        None.

    Params:
        None.
    """
    global running

    running = False

    if cam:
        try:
            cam.Stop()
            cam.Close()
        except Exception:
            pass


# Main GUI application class

class AstroCamApp:
    """
    Behaviour:
        Builds and runs the Tkinter camera control interface.

        The window has two main areas:
            1. Left side:
                Live video feed canvas.

            2. Right side:
                Gain slider.
                Gain number entry box.
                Exposure slider.
                Exposure number entry box.
                Camera status text.

    Effects:
        Creates Tkinter widgets.
        Starts the camera.
        Starts a repeating GUI refresh loop.

    Returns:
        None directly.
        An instance of this class represents the running GUI.

    Params:
        root (tk.Tk):
            The main Tkinter root window.
    """

    REFRESH_MS = 50
    # Number of milliseconds between GUI refreshes.
    # 50 ms is about 20 frames per second.

    FEED_W = 640
    # Width of the displayed camera feed canvas.

    FEED_H = 480
    # Height of the displayed camera feed canvas.

    def __init__(self, root):
        """
        Behaviour:
            Initialises the GUI application object.

            Stores the Tkinter root window, sets basic window properties,
            builds all widgets, starts the camera, and starts the refresh loop.

        Effects:
            Sets self.root and self.tk_image.
            Changes the root window title, background colour, and resize mode.
            Calls _build_ui().
            Calls _start_camera().
            Calls _schedule_refresh().

        Returns:
            None.

        Params:
            root (tk.Tk):
                The main Tkinter window created in main().
        """
        self.root = root
        self.tk_image = None

        self.root.title("ATR585M Camera Control")
        self.root.configure(bg="#0d0d1a")
        self.root.resizable(False, False)

        self._build_ui()
        self._start_camera()
        self._schedule_refresh()

    # Utility helpers

    def _clamp_int(self, value, min_value, max_value):
        """
        Behaviour:
            Converts a value into an integer and clamps it inside a safe range.

            This is used by the gain and exposure number entry boxes. It makes
            sure a typed value is a valid number and does not go below or above
            the allowed camera limits.

        Effects:
            Does not modify the camera or GUI directly.
            Only calculates and returns a safe integer value.

        Returns:
            int:
                The converted integer, limited to the allowed range.

            None:
                Returned if the value cannot be converted into a number.

        Params:
            value:
                The value to convert. This may come from a Tkinter variable,
                an entry box, or another part of the program.

            min_value (int):
                Lowest allowed value.

            max_value (int):
                Highest allowed value.
        """
        try:
            value = int(float(value))
        except (ValueError, TypeError, tk.TclError):
            return None

        return max(min_value, min(max_value, value))

    # Widget construction

    def _build_ui(self):
        """
        Behaviour:
            Builds the main two-column GUI layout.

            The left column contains the live camera feed.
            The right column contains the gain, exposure, and status controls.

        Effects:
            Creates the outer container frame.
            Calls _build_feed_panel().
            Calls _build_control_panel().

        Returns:
            None.

        Params:
            None.
        """
        container = tk.Frame(self.root, bg="#0d0d1a")
        container.pack(padx=12, pady=12, fill="both", expand=True)

        self._build_feed_panel(container)
        self._build_control_panel(container)

    def _build_feed_panel(self, parent):
        """
        Behaviour:
            Builds the live feed panel on the left side of the GUI.

            The panel contains a title label and a Tkinter canvas. The canvas
            is where the processed camera image is drawn during each refresh.

        Effects:
            Creates and packs feed_frame.
            Creates the "LIVE FEED" label.
            Creates self.feed_canvas.

        Returns:
            None.

        Params:
            parent (tk.Frame):
                Parent container that holds the feed panel.
        """
        feed_frame = tk.Frame(parent, bg="#0d0d1a")
        feed_frame.pack(side="left", padx=(0, 12))

        tk.Label(
            feed_frame,
            text="LIVE FEED",
            bg="#0d0d1a",
            fg="#00d4ff",
            font=("Consolas", 11, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.feed_canvas = tk.Canvas(
            feed_frame,
            width=self.FEED_W,
            height=self.FEED_H,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#1e3a5a"
        )
        self.feed_canvas.pack()

    def _build_control_panel(self, parent):
        """
        Behaviour:
            Builds the right-side control panel.

            This panel contains:
                1. Gain heading.
                2. Gain slider.
                3. Gain number entry box.
                4. Gain value label.
                5. Exposure heading.
                6. Exposure slider.
                7. Exposure number entry box.
                8. Exposure value label.
                9. Status heading.
                10. Live camera status label.

            The slider and number entry for each setting share the same
            Tkinter IntVar. This keeps the controls synchronised.

        Effects:
            Creates many Tkinter widgets.
            Sets the following instance variables:
                self.gain_var
                self.gain_slider
                self.gain_entry
                self.gain_lbl
                self.expo_var
                self.expo_slider
                self.expo_entry
                self.expo_lbl
                self.status_var

            Binds gain entry events to _on_gain_entry().
            Binds exposure entry events to _on_expo_entry().

        Returns:
            None.

        Params:
            parent (tk.Frame):
                Parent container that holds the control panel.
        """
        ctrl = tk.Frame(parent, bg="#0d0d1a")
        ctrl.pack(side="left", fill="y")

        lbl_h = dict(
            bg="#0d0d1a",
            fg="#00d4ff",
            font=("Consolas", 10, "bold")
        )

        lbl_v = dict(
            bg="#0d0d1a",
            fg="#e0e0e0",
            font=("Consolas", 10)
        )

        lbl_s = dict(
            bg="#0d0d1a",
            fg="#808090",
            font=("Consolas", 9)
        )

        # Gain control section
        #
        # Behaviour:
        #     Creates a gain slider and gain number entry box.
        #
        # Effects:
        #     Both controls use self.gain_var, so changing one updates the
        #     other.
        #
        # Returns:
        #     None.
        #
        # Params:
        #     None directly. Uses ctrl as the parent widget.
        

        tk.Label(ctrl, text="GAIN", **lbl_h).pack(anchor="w", pady=(0, 2))

        self.gain_var = tk.IntVar(value=DEFAULT_GAIN)

        gain_row = tk.Frame(ctrl, bg="#0d0d1a")
        gain_row.pack(fill="x", pady=(0, 2))

        self.gain_slider = tk.Scale(
            gain_row,
            variable=self.gain_var,
            from_=GAIN_MIN,
            to=GAIN_MAX,
            orient="horizontal",
            length=165,
            resolution=100,
            bg="#0d0d1a",
            fg="#e0e0e0",
            troughcolor="#1e1e3a",
            activebackground="#00d4ff",
            highlightthickness=0,
            showvalue=False,
            command=self._on_gain_change
        )
        self.gain_slider.pack(side="left", fill="x", expand=True)

        self.gain_entry = tk.Entry(
            gain_row,
            textvariable=self.gain_var,
            width=7,
            bg="#101020",
            fg="#e0e0e0",
            insertbackground="#e0e0e0",
            justify="right",
            font=("Consolas", 10)
        )
        self.gain_entry.pack(side="left", padx=(8, 0))

        self.gain_entry.bind("<Return>", self._on_gain_entry)
        self.gain_entry.bind("<FocusOut>", self._on_gain_entry)

        self.gain_lbl = tk.Label(
            ctrl,
            text=f"Value: {DEFAULT_GAIN}  ({DEFAULT_GAIN / 100:.1f}×)",
            **lbl_v
        )
        self.gain_lbl.pack(anchor="w", pady=(0, 12))

        # Exposure control section
        #
        # Behaviour:
        #     Creates an exposure slider and exposure number entry box.
        #
        # Effects:
        #     Both controls use self.expo_var, so changing one updates the
        #     other.
        #
        # Returns:
        #     None.
        #
        # Params:
        #     None directly. Uses ctrl as the parent widget.

        tk.Label(ctrl, text="EXPOSURE", **lbl_h).pack(anchor="w", pady=(0, 2))

        self.expo_var = tk.IntVar(value=DEFAULT_EXPOSURE_US)

        expo_row = tk.Frame(ctrl, bg="#0d0d1a")
        expo_row.pack(fill="x", pady=(0, 2))

        self.expo_slider = tk.Scale(
            expo_row,
            variable=self.expo_var,
            from_=EXPO_MIN_US,
            to=EXPO_MAX_US,
            orient="horizontal",
            length=165,
            resolution=100,
            bg="#0d0d1a",
            fg="#e0e0e0",
            troughcolor="#1e1e3a",
            activebackground="#00d4ff",
            highlightthickness=0,
            showvalue=False,
            command=self._on_expo_change
        )
        self.expo_slider.pack(side="left", fill="x", expand=True)

        self.expo_entry = tk.Entry(
            expo_row,
            textvariable=self.expo_var,
            width=10,
            bg="#101020",
            fg="#e0e0e0",
            insertbackground="#e0e0e0",
            justify="right",
            font=("Consolas", 10)
        )
        self.expo_entry.pack(side="left", padx=(8, 0))

        self.expo_entry.bind("<Return>", self._on_expo_entry)
        self.expo_entry.bind("<FocusOut>", self._on_expo_entry)

        self.expo_lbl = tk.Label(
            ctrl,
            text=f"Value: {DEFAULT_EXPOSURE_US} µs  ({DEFAULT_EXPOSURE_US / 1000:.1f} ms)",
            **lbl_v
        )
        self.expo_lbl.pack(anchor="w", pady=(0, 20))

        # Status section
        #
        # Behaviour:
        #     Creates a text area that shows live camera readback values.
        #
        # Effects:
        #     Creates self.status_var and binds it to a Tkinter Label.
        #
        # Returns:
        #     None.
        #
        # Params:
        #     None directly. Uses ctrl as the parent widget.

        tk.Label(ctrl, text="STATUS", **lbl_h).pack(anchor="w", pady=(0, 4))

        self.status_var = tk.StringVar(value="Connecting...")

        tk.Label(
            ctrl,
            textvariable=self.status_var,
            wraplength=240,
            justify="left",
            **lbl_s
        ).pack(anchor="w")

    # Camera startup

    def _start_camera(self):
        """
        Behaviour:
            Attempts to open and start the camera.

            If the camera opens successfully, the status label is updated with
            the camera resolution and default settings.

            If no camera is found, the GUI stays open and displays a warning
            message instead of crashing.

        Effects:
            Calls open_camera().
            Updates self.status_var.

        Returns:
            None.

        Params:
            None.
        """
        ok = open_camera()

        if ok:
            self.status_var.set(
                f"Camera: {WIDTH}×{HEIGHT}\n"
                f"Gain: {DEFAULT_GAIN}\n"
                f"Exp: {DEFAULT_EXPOSURE_US} µs\n"
                f"TEC: {DEFAULT_TEMP} °C"
            )
        else:
            self.status_var.set("No camera found.\nCheck USB & power.")

    # Slider callbacks

    def _on_gain_change(self, val):
        """
        Behaviour:
            Runs whenever the gain slider is moved.

            Converts the slider value into an integer, sends it to the camera,
            updates the gain entry box, and updates the gain readout label.

        Effects:
            Calls cam.put_ExpoAGain() if the camera is connected.
            Updates self.gain_var.
            Updates self.gain_lbl.

        Returns:
            None.

        Params:
            val (str | int | float):
                Current gain value from the slider.
                Tkinter usually sends this as a string.
        """
        gain = int(float(val))

        if cam:
            try:
                cam.put_ExpoAGain(gain)
            except Exception as e:
                print(f"[Gain error] {e}")

        self.gain_var.set(gain)

        self.gain_lbl.config(
            text=f"Value: {gain}  ({gain / 100:.1f}×)"
        )

    def _on_expo_change(self, val):
        """
        Behaviour:
            Runs whenever the exposure slider is moved.

            Converts the slider value into an integer microsecond value, sends
            it to the camera, updates the exposure entry box, and updates the
            exposure readout label.

        Effects:
            Calls cam.put_ExpoTime() if the camera is connected.
            Updates self.expo_var.
            Updates self.expo_lbl.

        Returns:
            None.

        Params:
            val (str | int | float):
                Current exposure value from the slider in microseconds.
                Tkinter usually sends this as a string.
        """
        us = int(float(val))

        if cam:
            try:
                cam.put_ExpoTime(us)
            except Exception as e:
                print(f"[Exposure error] {e}")

        self.expo_var.set(us)

        self.expo_lbl.config(
            text=f"Value: {us} µs  ({us / 1000:.1f} ms)"
        )

    # Entry box callbacks

    def _on_gain_entry(self, event=None):
        """
        Behaviour:
            Runs when the user presses Enter in the gain number box or clicks
            away from it.

            Reads the typed gain value, checks whether it is valid, clamps it
            inside the allowed gain range, snaps it to the nearest slider step,
            and applies it to the camera.

        Effects:
            Updates self.gain_var.
            Updates the gain slider because it shares self.gain_var.
            Calls _on_gain_change(), which updates the camera and label.

        Returns:
            None.

        Params:
            event (tk.Event | None):
                Tkinter event object produced by <Return> or <FocusOut>.
                It is not directly used.
        """
        gain = self._clamp_int(
            self.gain_var.get(),
            GAIN_MIN,
            GAIN_MAX
        )

        if gain is None:
            gain = DEFAULT_GAIN

        gain = round(gain / 100) * 100

        gain = self._clamp_int(
            gain,
            GAIN_MIN,
            GAIN_MAX
        )

        self.gain_var.set(gain)
        self._on_gain_change(gain)

    def _on_expo_entry(self, event=None):
        """
        Behaviour:
            Runs when the user presses Enter in the exposure number box or
            clicks away from it.

            Reads the typed exposure value, checks whether it is valid, clamps
            it inside the allowed exposure range, snaps it to the nearest
            slider step, and applies it to the camera.

        Effects:
            Updates self.expo_var.
            Updates the exposure slider because it shares self.expo_var.
            Calls _on_expo_change(), which updates the camera and label.

        Returns:
            None.

        Params:
            event (tk.Event | None):
                Tkinter event object produced by <Return> or <FocusOut>.
                It is not directly used.
        """
        us = self._clamp_int(
            self.expo_var.get(),
            EXPO_MIN_US,
            EXPO_MAX_US
        )

        if us is None:
            us = DEFAULT_EXPOSURE_US

        us = round(us / 100) * 100

        us = self._clamp_int(
            us,
            EXPO_MIN_US,
            EXPO_MAX_US
        )

        self.expo_var.set(us)
        self._on_expo_change(us)

    # Refresh loop

    def _schedule_refresh(self):
        """
        Behaviour:
            Starts the repeating GUI refresh loop.

            This schedules _refresh() to run after REFRESH_MS milliseconds.
            The _refresh() method then schedules itself again, creating a
            continuous update loop.

        Effects:
            Calls self.root.after().

        Returns:
            None.

        Params:
            None.
        """
        self.root.after(self.REFRESH_MS, self._refresh)

    def _refresh(self):
        """
        Behaviour:
            Updates the video feed and camera status.

            The refresh process:
                1. Safely copies the newest camera frame.
                2. Normalises the 16-bit image into an 8-bit image.
                3. Applies a false-colour JET colour map.
                4. Resizes the image to fit the Tkinter canvas.
                5. Converts the image into a Tkinter PhotoImage.
                6. Draws the image on the canvas.
                7. Reads live gain, exposure, and temperature from the camera.
                8. Updates the status label.
                9. Schedules the next refresh.

        Effects:
            Reads latest_raw under frame_lock.
            Updates self.tk_image.
            Draws onto self.feed_canvas.
            Updates self.status_var.
            Calls self.root.after() to schedule the next refresh.

        Returns:
            None.

        Params:
            None.
        """
        with frame_lock:
            frame = latest_raw.copy() if latest_raw is not None else None

        if frame is not None:
            norm = cv2.normalize(
                frame,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            colour = cv2.applyColorMap(
                norm,
                cv2.COLORMAP_JET
            )

            resized = cv2.resize(
                colour,
                (self.FEED_W, self.FEED_H),
                interpolation=cv2.INTER_LINEAR
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            h, w = rgb.shape[:2]

            ppm = (
                f"P6\n{w} {h}\n255\n".encode()
                + rgb.tobytes()
            )

            self.tk_image = tk.PhotoImage(data=ppm)

            self.feed_canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.tk_image
            )

        if cam and running:
            try:
                exp = cam.get_ExpoTime()
                gain = cam.get_ExpoAGain()
                temp = cam.get_Temperature() / 10

                self.status_var.set(
                    f"Gain : {gain}  ({gain / 100:.1f}×)\n"
                    f"Exp  : {exp} µs  ({exp / 1000:.1f} ms)\n"
                    f"TEC  : {temp:.1f} °C"
                )

            except Exception:
                pass

        self.root.after(self.REFRESH_MS, self._refresh)

    
    # Shutdown

    def on_close(self):
        """
        Behaviour:
            Handles the user closing the GUI window.

            Stops the camera before destroying the Tkinter window so the SDK
            callback does not continue after the GUI has closed.

        Effects:
            Calls close_camera().
            Calls self.root.destroy().

        Returns:
            None.

        Params:
            None.
        """
        close_camera()
        self.root.destroy()


# Program entry point

def main():
    """
    Behaviour:
        Creates the Tkinter root window, builds the AstroCamApp object, binds
        the window close button to a safe shutdown method, and starts the
        Tkinter event loop.

    Effects:
        Creates the main GUI window.
        Instantiates AstroCamApp.
        Registers app.on_close as the close-window handler.
        Starts root.mainloop(), which blocks until the GUI is closed.

    Returns:
        None.

    Params:
        None.
    """
    root = tk.Tk()

    app = AstroCamApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.on_close
    )

    root.mainloop()


if __name__ == "__main__":
    main()