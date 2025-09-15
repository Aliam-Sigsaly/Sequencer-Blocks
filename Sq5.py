import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
from collections import deque
import sounddevice as sd
import math
import queue

class EnvelopeSequencer:
    def __init__(self, root):
        self.root = root
        self.root.title("Envelope Sequencer with High-Precision Clock")
        self.root.geometry("900x900")

        # High-precision clock variables
        self.sample_rate = 40000
        self.is_playing = False
        self.stop_counter = False
        self.counter_value = 0
        self.note_start_sample = 0
        self.current_note = 0
        self.sample_count = 0
        self.start_time = 0
        self.expected_samples = 0
        self.last_time = time.perf_counter()
        self.normalized_buffer = deque(maxlen=100)

        # Envelope variables
        self.attack_val = tk.DoubleVar(value=20)
        self.decay_val = tk.DoubleVar(value=100)
        self.norm_attack = tk.StringVar(value="0.00")
        self.norm_decay = tk.StringVar(value="0.00")
        self.amplitude = tk.StringVar(value="0.00")

        # Reference shape
        self.ref_attack = 20.0
        self.ref_decay = 100.0
        self.show_reference = False

        # Shape recording and storage
        self.actual_shape = []
        self.recording = False
        self.saved_shapes = []
        self.shape_colors = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF",
                            "#FFFF00", "#00FFFF", "#FF8800", "#8800FF"]

        # Sequencer variables
        self.sequencer_notes = [tk.DoubleVar(value=20000) for _ in range(4)]
        self.current_note_index = -1

        # Audio variables
        self.audio_queue = queue.Queue()
        self.buffer_size = 40000
        self.is_audio_playing = False
        self.stop_audio = False
        self.audio_thread = None
        self.freq_var = tk.DoubleVar(value=440.0)
        self.audio_stream = None

        # Jam parameters
        self.note_durations = [10000, 20000, 15000, 25000]  # Four note durations
        self.note_duration_vars = [tk.IntVar(value=d) for d in self.note_durations]

        self.create_widgets()
        self.setup_audio_buffers()

    def setup_audio_buffers(self):
        self.normalized_buffer = deque(maxlen=100)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Envelope canvas
        self.canvas = tk.Canvas(main_frame, width=400, height=200, bg='white', relief='sunken', bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        # Attack Control
        ttk.Label(main_frame, text="Attack Ratio (%):").grid(row=1, column=0, padx=10, pady=5)
        self.attack_spin = ttk.Spinbox(main_frame, from_=0, to=100, width=8, textvariable=self.attack_val, command=self.update_envelope)
        self.attack_spin.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Decay Control
        ttk.Label(main_frame, text="Decay Ratio (%):").grid(row=2, column=0, padx=10, pady=5)
        self.decay_spin = ttk.Spinbox(main_frame, from_=0, to=100, width=8, textvariable=self.decay_val, command=self.update_envelope)
        self.decay_spin.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Normalized Values
        ttk.Label(main_frame, text="Normalized Attack:").grid(row=3, column=0, padx=10, pady=5)
        ttk.Label(main_frame, textvariable=self.norm_attack, width=8, relief="solid").grid(row=3, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(main_frame, text="Normalized Decay:").grid(row=4, column=0, padx=10, pady=5)
        ttk.Label(main_frame, textvariable=self.norm_decay, width=8, relief="solid").grid(row=4, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(main_frame, text="Current Amplitude:").grid(row=5, column=0, padx=10, pady=5)
        ttk.Label(main_frame, textvariable=self.amplitude, width=8, relief="solid").grid(row=5, column=1, padx=10, pady=5, sticky="w")

        # Audio controls
        audio_frame = ttk.Frame(main_frame)
        audio_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(audio_frame, text="Frequency:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(audio_frame, from_=50, to=2000, width=8, textvariable=self.freq_var).pack(side=tk.LEFT, padx=5)

        # Jam controls for note durations
        jam_frame = ttk.LabelFrame(main_frame, text="Jam Controls", padding="5")
        jam_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        for i, var in enumerate(self.note_duration_vars):
            ttk.Label(jam_frame, text=f"Note {i+1} duration (samples):").grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            spin = ttk.Spinbox(jam_frame, from_=1000, to=100000, increment=1000,
                              textvariable=var, width=10)
            spin.grid(row=i, column=1, padx=5, pady=2)
            spin.bind('<Return>', self.update_note_durations)
            spin.bind('<FocusOut>', self.update_note_durations)

        # Sequencer UI
        seq_frame = ttk.LabelFrame(main_frame, text="Sequencer", padding="5")
        seq_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Note indicators canvas
        self.note_canvas = tk.Canvas(seq_frame, width=400, height=50, bg='white', relief='sunken', bd=2)
        self.note_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Timing diagnostics
        ttk.Label(main_frame, text="Actual Sample Rate:").grid(row=9, column=0, pady=5, sticky=tk.W)
        self.rate_var = tk.StringVar(value="0 Hz")
        ttk.Label(main_frame, textvariable=self.rate_var).grid(row=9, column=1, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Expected Sample Rate:").grid(row=10, column=0, pady=5, sticky=tk.W)
        self.expected_rate_var = tk.StringVar(value=f"{self.sample_rate} Hz")
        ttk.Label(main_frame, textvariable=self.expected_rate_var).grid(row=10, column=1, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Timing Accuracy:").grid(row=11, column=0, pady=5, sticky=tk.W)
        self.accuracy_var = tk.StringVar(value="100%")
        ttk.Label(main_frame, textvariable=self.accuracy_var).grid(row=11, column=1, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Normalized Time:").grid(row=12, column=0, pady=5, sticky=tk.W)
        self.normalized_var = tk.StringVar(value="0.0000")
        ttk.Label(main_frame, textvariable=self.normalized_var).grid(row=12, column=1, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Current Note:").grid(row=13, column=0, pady=5, sticky=tk.W)
        self.note_var = tk.StringVar(value="1")
        ttk.Label(main_frame, textvariable=self.note_var).grid(row=13, column=1, pady=5, sticky=tk.W)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=14, column=0, columnspan=2, pady=10, sticky="ew")

        self.play_btn = ttk.Button(control_frame, text="Start Sequence", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.ref_btn = ttk.Button(control_frame, text="Set Reference Shape", command=self.set_reference_shape)
        self.ref_btn.pack(side=tk.LEFT, padx=5)

        self.toggle_ref_btn = ttk.Button(control_frame, text="Show Reference", command=self.toggle_reference)
        self.toggle_ref_btn.pack(side=tk.LEFT, padx=5)

        self.record_btn = ttk.Button(control_frame, text="Record Shape", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Clear All Shapes", command=self.clear_all_shapes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Shape", command=self.save_shape_and_update).pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Initialize
        self.update_envelope()
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.draw_note_indicators()

    def samples_to_ms(self, samples):
        """Convert samples to milliseconds"""
        return samples * 1000 / self.sample_rate

    def draw_note_indicators(self):
        """Draw the note indicators on the sequencer canvas"""
        self.note_canvas.delete("all")
        width = self.note_canvas.winfo_width()
        height = self.note_canvas.winfo_height()

        if width <= 1 or height <= 1:  # Canvas not yet rendered
            return

        # Draw 4 rectangles for the notes
        rect_width = (width - 50) / 4
        for i in range(4):
            x1 = 10 + i * (rect_width + 10)
            y1 = 10
            x2 = x1 + rect_width
            y2 = height - 10

            # Color red if this is the current note, otherwise light gray
            color = "red" if i == self.current_note else "light gray"
            self.note_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

            # Add note number above the rectangle
            self.note_canvas.create_text((x1 + x2) / 2, 5,
                                        text=f"{i+1}", anchor="n", font=("Arial", 10, "bold"))

            # Add note duration text in ms
            note_ms = self.samples_to_ms(self.note_durations[i])
            self.note_canvas.create_text((x1 + x2) / 2, height / 2,
                                        text=f"{note_ms:.1f}ms")

    def start_audio_playback(self):
        """Start audio playback in a separate thread"""
        if self.is_audio_playing:
            return

        self.is_audio_playing = True
        self.stop_audio = False

        # Start audio in a separate thread to avoid blocking the GUI
        self.audio_thread = threading.Thread(target=self.audio_playback_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_audio_playback(self):
        """Stop audio playback"""
        self.stop_audio = True
        self.is_audio_playing = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    def audio_playback_loop(self):
        """Audio playback loop running in a separate thread"""
        try:
            # Open audio stream
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.audio_callback
            )
            self.audio_stream.start()

            # Keep the thread alive while audio is playing
            while not self.stop_audio:
                time.sleep(0.1)

        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            self.is_audio_playing = False

    def audio_callback(self, outdata, frames, time_info, status):
        """Audio callback function that generates sound in real-time"""
        if status:
            print(f"Audio status: {status}")

        # Generate audio data for the requested number of frames
        data = np.zeros(frames, dtype=np.float32)

        for i in range(frames):
            if not self.is_playing:
                break

            # Calculate the current time in the note
            elapsed_in_note = self.counter_value - self.note_start_sample
            current_note_duration = self.note_durations[self.current_note]
            normalized_value = elapsed_in_note / current_note_duration if current_note_duration > 0 else 0

            # Calculate envelope
            t_percent = normalized_value * 100
            A = self.attack_val.get()
            D = max(self.decay_val.get(), A)
            amp = self.compute_amplitude(t_percent, A, D)

            # Generate sine wave with current frequency and envelope
            frequency = self.freq_var.get()
            time_in_seconds = self.sample_count / self.sample_rate
            data[i] = amp * math.sin(2 * math.pi * frequency * time_in_seconds)

            # Increment counters
            self.counter_value += 1
            self.sample_count += 1

            # Check if we've reached the end of the current note
            if elapsed_in_note >= current_note_duration:
                # Move to next note
                self.note_start_sample = self.counter_value
                self.current_note = (self.current_note + 1) % len(self.note_durations)
                # Update GUI
                self.root.after(0, self.update_display)

        # Ensure the data is in the correct format
        outdata[:] = data.reshape(-1, 1)

    def update_note_durations(self, event=None):
        for i, var in enumerate(self.note_duration_vars):
            try:
                self.note_durations[i] = max(1000, var.get())
            except tk.TclError:
                pass
        self.draw_note_indicators()

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.is_playing:
            return

        self.is_playing = True
        self.stop_counter = False
        self.play_btn.config(text="Stop Sequence")
        self.counter_value = 0
        self.note_start_sample = 0
        self.current_note = 0
        self.sample_count = 0
        self.start_time = time.time()

        # Start audio playback
        self.start_audio_playback()

        # Start counter in a separate thread
        self.counter_thread = threading.Thread(target=self.counter_loop)
        self.counter_thread.daemon = True
        self.counter_thread.start()

    def stop_playback(self):
        self.stop_counter = True
        self.is_playing = False
        self.play_btn.config(text="Start Sequence")
        self.stop_audio_playback()

        # Calculate actual sample rate
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            actual_rate = self.sample_count / elapsed
            self.rate_var.set(f"{actual_rate:.0f} Hz")

    def counter_loop(self):
        sample_interval = 1.0 / self.sample_rate
        last_update_time = time.time()
        update_interval = 0.05  # Update GUI every 50ms
        next_time = time.perf_counter()
        samples_per_update = int(self.sample_rate * update_interval)

        # Pre-allocate arrays for efficiency
        normalized_chunk = np.zeros(samples_per_update)

        while not self.stop_counter:
            # Process a chunk of samples for efficiency
            for i in range(samples_per_update):
                # Increment counter
                self.counter_value += 1
                self.sample_count += 1

                # Check if we've reached the end of the current note
                current_note_duration = self.note_durations[self.current_note]
                elapsed_in_note = self.counter_value - self.note_start_sample

                if elapsed_in_note >= current_note_duration:
                    # Move to next note
                    self.note_start_sample = self.counter_value
                    self.current_note = (self.current_note + 1) % len(self.note_durations)
                    elapsed_in_note = 0
                    current_note_duration = self.note_durations[self.current_note]

                    # Clear previous step recording shape when new note is triggered
                    if self.recording:
                        self.actual_shape = []

                # Calculate normalized value
                normalized_value = elapsed_in_note / current_note_duration
                normalized_chunk[i] = normalized_value

                # Record shape if recording
                if self.recording:
                    t_percent = normalized_value * 100
                    A = self.attack_val.get()
                    D = max(self.decay_val.get(), A)
                    amp = self.compute_amplitude(t_percent, A, D)
                    self.actual_shape.append((t_percent, amp))

            # Add the chunk to the buffer
            self.normalized_buffer.extend(normalized_chunk)

            # Update GUI
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                self.root.after(0, self.update_display)
                last_update_time = current_time

            # High-precision timing
            next_time += update_interval
            current_time = time.perf_counter()
            sleep_time = next_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're running behind schedule
                next_time = current_time

    def update_display(self):
        # Update all display elements
        if self.normalized_buffer:
            normalized_value = self.normalized_buffer[-1]
            self.normalized_var.set(f"{normalized_value:.4f}")

            # Update envelope display
            A = self.attack_val.get()
            D = max(self.decay_val.get(), A)

            # Calculate amplitude and normalized values
            t_percent = normalized_value * 100
            if A > 0 and t_percent <= A:
                norm_a = t_percent / A
                self.norm_attack.set(f"{norm_a:.2f}")
                self.norm_decay.set("0.00")
                current_amp = norm_a
            elif D > A and A <= t_percent <= D:
                norm_d = (t_percent - A) / (D - A)
                self.norm_attack.set("0.00")
                self.norm_decay.set(f"{norm_d:.2f}")
                current_amp = 1.0 - norm_d
            else:
                self.norm_attack.set("0.00")
                self.norm_decay.set("0.00")
                current_amp = 0.0

            self.amplitude.set(f"{current_amp:.2f}")
            self.draw_envelope(A, D, t_percent)

        self.note_var.set(str(self.current_note + 1))
        self.draw_note_indicators()

        # Update actual sample rate and timing accuracy
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            actual_rate = self.sample_count / elapsed
            self.rate_var.set(f"{actual_rate:.0f} Hz")

            # Calculate timing accuracy
            expected_samples = elapsed * self.sample_rate
            accuracy = (self.sample_count / expected_samples) * 100 if expected_samples > 0 else 0
            self.accuracy_var.set(f"{accuracy:.1f}%")

    def set_reference_shape(self):
        self.ref_attack = self.attack_val.get()
        self.ref_decay = self.decay_val.get()
        self.update_envelope()

    def toggle_reference(self):
        self.show_reference = not self.show_reference
        self.toggle_ref_btn.config(text="Hide Reference" if self.show_reference else "Show Reference")
        self.update_envelope()

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.record_btn.config(text="Record Shape")
        else:
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.actual_shape = []  # Reset recording buffer

    def save_shape_and_update(self):
        if self.actual_shape:
            self.saved_shapes.append(self.actual_shape[:])
            if len(self.saved_shapes) > 8:
                self.saved_shapes.pop(0)
            print(f"Shape saved. Total shapes: {len(self.saved_shapes)}")

    def clear_all_shapes(self):
        self.saved_shapes = []
        self.update_envelope()

    def on_canvas_resize(self, event):
        self.update_envelope()

    def compute_amplitude(self, t, A, D):
        if A > 0 and t <= A:
            return t / A
        elif D > A and A <= t <= D:
            return 1.0 - (t - A) / (D - A)
        return 0.0

    def draw_envelope(self, A, D, t_percent):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        padding = 20
        draw_width = width - 2 * padding
        draw_height = height - 2 * padding

        def t_to_x(t_val):
            return padding + (t_val / 100) * draw_width

        def a_to_y(amplitude):
            return padding + (1 - amplitude) * draw_height

        # Draw axes and grid
        self.canvas.create_line(padding, height - padding, width - padding, height - padding, width=2)
        self.canvas.create_line(padding, padding, padding, height - padding, width=2)
        for i in range(1, 10):
            x = padding + i * draw_width / 10
            self.canvas.create_line(x, padding, x, height - padding, dash=(2, 2), fill="gray")
            y = padding + i * draw_height / 10
            self.canvas.create_line(padding, y, width - padding, y, dash=(2, 2), fill="gray")

        # Draw saved shapes
        for idx, shape in enumerate(self.saved_shapes):
            if shape and len(shape) >= 2:
                points = []
                color = self.shape_colors[idx % len(self.shape_colors)]
                for time_val, amp_val in shape:
                    points.append(t_to_x(time_val))
                    points.append(a_to_y(amp_val))
                self.canvas.create_line(points, fill=color, width=1, smooth=True)

        # Draw reference shape
        if self.show_reference:
            ref_start = (padding, height - padding)
            ref_peak = (t_to_x(self.ref_attack), padding)
            ref_end = (t_to_x(self.ref_decay), height - padding)
            self.canvas.create_line(ref_start[0], ref_start[1], ref_peak[0], ref_peak[1],
                                   fill="#777777", width=1, dash=(4, 4))
            self.canvas.create_line(ref_peak[0], ref_peak[1], ref_end[0], ref_end[1],
                                   fill="#777777", width=1, dash=(4, 4))

        # Draw current recorded shape (if recording and has points)
        if self.recording and self.actual_shape and len(self.actual_shape) >= 2:
            points = []
            for time_val, amp_val in self.actual_shape:
                points.append(t_to_x(time_val))
                points.append(a_to_y(amp_val))
            self.canvas.create_line(points, fill="red", width=1.5, smooth=True)

        # Draw current envelope
        start_point = (padding, height - padding)
        peak_point = (t_to_x(A), padding)
        end_point = (t_to_x(D), height - padding)
        self.canvas.create_line(start_point[0], start_point[1], peak_point[0], peak_point[1],
                               fill="#8888FF", width=1)
        self.canvas.create_line(peak_point[0], peak_point[1], end_point[0], end_point[1],
                               fill="#8888FF", width=1)

        # Draw control points
        self.canvas.create_oval(peak_point[0]-5, peak_point[1]-5, peak_point[0]+5, peak_point[1]+5, fill="red")
        self.canvas.create_oval(end_point[0]-5, end_point[1]-5, end_point[0]+5, end_point[1]+5, fill="green")

        # Draw current time marker
        if t_percent >= 0:
            current_x = t_to_x(t_percent)
            current_amp = self.compute_amplitude(t_percent, A, D)
            current_y = a_to_y(current_amp)
            self.canvas.create_line(current_x, padding, current_x, height - padding, dash=(2, 2), fill="purple")
            self.canvas.create_oval(current_x-6, current_y-6, current_x+6, current_y+6, fill="gold", outline="black")
            self.canvas.create_text(current_x, current_y - 15, text=f"t={t_percent:.1f}%\na={current_amp:.2f}",
                                   anchor="s", fill="darkred")

    def update_envelope(self, *args):
        try:
            A = self.attack_val.get()
            D = max(self.decay_val.get(), A)  # Ensure D >= A

            # Get the current normalized time if playing, otherwise use 0
            if self.normalized_buffer:
                t_percent = self.normalized_buffer[-1] * 100
            else:
                t_percent = 0

            # Calculate amplitude and normalized values
            if A > 0 and t_percent <= A:
                norm_a = t_percent / A
                self.norm_attack.set(f"{norm_a:.2f}")
                self.norm_decay.set("0.00")
                current_amp = norm_a
            elif D > A and A <= t_percent <= D:
                norm_d = (t_percent - A) / (D - A)
                self.norm_attack.set("0.00")
                self.norm_decay.set(f"{norm_d:.2f}")
                current_amp = 1.0 - norm_d
            else:
                self.norm_attack.set("0.00")
                self.norm_decay.set("0.00")
                current_amp = 0.0

            self.amplitude.set(f"{current_amp:.2f}")
            self.draw_envelope(A, D, t_percent)

        except (ValueError, ZeroDivisionError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = EnvelopeSequencer(root)
    root.mainloop()
