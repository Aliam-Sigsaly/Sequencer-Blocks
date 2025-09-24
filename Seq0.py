import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import threading
import time

class EnvelopeSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Envelope Simulator with Sequencer")
        self.root.geometry("800x900")
        self.audio_buffer = []
        self.buffer_size = 40000

        # Variables
        self.time_val = tk.DoubleVar(value=0)
        self.attack_val = tk.DoubleVar(value=20)
        self.decay_val = tk.DoubleVar(value=100)
        self.norm_attack = tk.StringVar(value="0.00")
        self.norm_decay = tk.StringVar(value="0.00")
        self.amplitude = tk.StringVar(value="0.00")

        # Audio playback variables
        self.sample_rate = 40000  # 40kHz sample rate
        self.is_playing = False
        self.audio_thread = None
        self.stop_audio = False

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

        # Clock variables
        self.clock_running = False
        self.after_id = None
        self.updating_from_clock = False

        # Canvas for visualization
        self.canvas = tk.Canvas(root, width=400, height=200, bg='white', relief='sunken', bd=2)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        # Rate Time Spinbox - now in samples
        ttk.Label(root, text="Rate Time (samples):").grid(row=1, column=0, padx=10, pady=5)
        self.rate_time_spin = ttk.Spinbox(root, from_=0, to=self.sample_rate*2, width=8,
                                         textvariable=self.time_val, command=self.update_envelope)
        self.rate_time_spin.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Attack Control
        ttk.Label(root, text="Attack Ratio (%):").grid(row=2, column=0, padx=10, pady=5)
        self.attack_spin = ttk.Spinbox(root, from_=0, to=100, width=8, textvariable=self.attack_val, command=self.update_envelope)
        self.attack_spin.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Decay Control
        ttk.Label(root, text="Decay Ratio (%):").grid(row=3, column=0, padx=10, pady=5)
        self.decay_spin = ttk.Spinbox(root, from_=0, to=100, width=8, textvariable=self.decay_val, command=self.update_envelope)
        self.decay_spin.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # Normalized Values
        ttk.Label(root, text="Normalized Attack:").grid(row=4, column=0, padx=10, pady=5)
        ttk.Label(root, textvariable=self.norm_attack, width=8, relief="solid").grid(row=4, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Normalized Decay:").grid(row=5, column=0, padx=10, pady=5)
        ttk.Label(root, textvariable=self.norm_decay, width=8, relief="solid").grid(row=5, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Current Amplitude:").grid(row=6, column=0, padx=10, pady=5)
        ttk.Label(root, textvariable=self.amplitude, width=8, relief="solid").grid(row=6, column=1, padx=10, pady=5, sticky="w")

        # Audio playback controls
        audio_frame = ttk.Frame(root)
        audio_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        ttk.Label(audio_frame, text="Audio Playback:").pack(side=tk.LEFT, padx=(0, 10))
        self.play_audio_btn = ttk.Button(audio_frame, text="Play Audio", command=self.toggle_audio_playback)
        self.play_audio_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(audio_frame, text="Frequency:").pack(side=tk.LEFT, padx=(20, 5))
        self.freq_var = tk.DoubleVar(value=440.0)
        ttk.Spinbox(audio_frame, from_=50, to=2000, width=8, textvariable=self.freq_var).pack(side=tk.LEFT, padx=5)

        # Control buttons
        control_frame = ttk.Frame(root)
        control_frame.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

        # Clock controls (simplified without JamB)
        clock_frame = ttk.Frame(control_frame)
        clock_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.play_btn = ttk.Button(clock_frame, text="Play", command=self.toggle_clock)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        # Shape controls
        shape_frame = ttk.Frame(control_frame)
        shape_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.ref_btn = ttk.Button(shape_frame, text="Set Reference Shape", command=self.set_reference_shape)
        self.ref_btn.pack(side=tk.LEFT, padx=5)
        self.toggle_ref_btn = ttk.Button(shape_frame, text="Show Reference", command=self.toggle_reference)
        self.toggle_ref_btn.pack(side=tk.LEFT, padx=5)
        self.record_btn = ttk.Button(shape_frame, text="Record Shape", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Multi-envelope controls
        multi_frame = ttk.Frame(control_frame)
        multi_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(multi_frame, text="Clear All Shapes", command=self.clear_all_shapes).pack(side=tk.LEFT, padx=5)
        ttk.Button(multi_frame, text="Save Shape", command=self.save_shape_and_update).pack(side=tk.LEFT, padx=5)

        # Sequencer variables - now in samples
        self.sequencer_notes = [tk.DoubleVar(value=20000) for _ in range(4)]
        self.sequencer_running = False
        self.sequencer_thread = None
        self.current_note_index = -1

        # Create sequencer UI
        self.create_sequencer_ui()

        # Initialize
        self.time_val.trace_add('write', self.on_time_val_changed)
        self.update_envelope()
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Ensure audio is stopped when window closes
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_audio_playback(self):
        """Toggle audio playback"""
        if self.is_playing:
            self.stop_audio_playback()
        else:
            self.start_audio_playback()

    def start_audio_playback(self):
        """Start audio playback in a separate thread"""
        if self.is_playing:
            return

        self.is_playing = True
        self.stop_audio = False
        self.play_audio_btn.config(text="Stop Audio")

        # Start audio in a separate thread to avoid blocking the GUI
        self.audio_thread = threading.Thread(target=self.audio_playback_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_audio_playback(self):
        """Stop audio playback"""
        self.stop_audio = True
        self.is_playing = False
        self.play_audio_btn.config(text="Play Audio")

    def audio_playback_loop(self):
        """Audio playback loop running in a separate thread"""
        try:
            # Calculate total duration based on decay time
            total_duration = self.decay_val.get() / 100.0 * 2.0

            # Generate time array for the entire envelope
            t_audio = np.linspace(0, total_duration, int(total_duration * self.sample_rate), False)

            # Generate carrier frequency
            frequency = self.freq_var.get()
            carrier = np.sin(2 * np.pi * frequency * t_audio)

            # Apply envelope to carrier
            A = self.attack_val.get()
            D = self.decay_val.get()

            # Calculate envelope for each sample
            envelope = np.zeros_like(t_audio)
            for i, t in enumerate(t_audio):
                # Convert time to our 0-100 scale
                t_scaled = t / total_duration * 100
                envelope[i] = self.compute_amplitude(t_scaled, A, D)

            # Apply envelope to carrier
            audio_signal = carrier * envelope

            # Play audio with streaming
            with sd.OutputStream(samplerate=self.sample_rate, channels=1) as stream:
                # Split audio into chunks for responsive stopping
                chunk_size = 1024
                for i in range(0, len(audio_signal), chunk_size):
                    if self.stop_audio:
                        break
                    chunk = audio_signal[i:i+chunk_size]

                    # Store in buffer
                    self.audio_buffer.extend(chunk)
                    if len(self.audio_buffer) > self.buffer_size:
                        self.audio_buffer = self.audio_buffer[-self.buffer_size:]

                    stream.write(chunk.reshape(-1, 1))

        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            self.is_playing = False
            self.root.after(100, lambda: self.play_audio_btn.config(text="Play Audio"))

    def create_sequencer_ui(self):
        """Create sequencer controls"""
        sequencer_frame = ttk.LabelFrame(self.root, text="Sequencer (Keys)")
        sequencer_frame.grid(row=9, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Note duration controls - now in samples
        for i in range(4):
            ttk.Label(sequencer_frame, text=f"Note {i+1} (samples):").grid(row=0, column=i*2, padx=5, pady=5)
            spinbox = ttk.Spinbox(sequencer_frame, from_=10, to=20000, width=8,
                                textvariable=self.sequencer_notes[i],
                                command=self.update_note_display)
            spinbox.grid(row=0, column=i*2+1, padx=5, pady=5)

        # Canvas for note indicators
        self.note_canvas = tk.Canvas(sequencer_frame, width=400, height=50, bg='white', relief='sunken', bd=2)
        self.note_canvas.grid(row=1, column=0, columnspan=8, padx=10, pady=10, sticky="ew")

        # Draw initial note indicators
        self.draw_note_indicators()

        # Sequencer controls
        control_frame = ttk.Frame(sequencer_frame)
        control_frame.grid(row=2, column=0, columnspan=8, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Sequencer", command=self.toggle_sequencer)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Play Note", command=self.play_current_note).pack(side=tk.LEFT, padx=5)

    def update_note_display(self):
        """Update the note display when values change"""
        self.draw_note_indicators()

    def samples_to_ms(self, samples):
        """Convert samples to milliseconds"""
        return samples * 1000 / self.sample_rate

    def draw_note_indicators(self):
        """Draw the note indicators on the sequencer canvas"""
        self.note_canvas.delete("all")
        width = self.note_canvas.winfo_width()
        height = self.note_canvas.winfo_height()

        # Draw 4 rectangles for the notes
        rect_width = (width - 50) / 4
        for i in range(4):
            x1 = 10 + i * (rect_width + 10)
            y1 = 10
            x2 = x1 + rect_width
            y2 = height - 10

            # Color red if this is the current note, otherwise light gray
            color = "red" if i == self.current_note_index else "light gray"
            self.note_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

            # Add note number above the rectangle
            self.note_canvas.create_text((x1 + x2) / 2, 5,
                                        text=f"{i+1}", anchor="n", font=("Arial", 10, "bold"))

            # Add note duration text in ms
            note_ms = self.samples_to_ms(self.sequencer_notes[i].get())
            self.note_canvas.create_text((x1 + x2) / 2, height / 2,
                                        text=f"{note_ms:.1f}ms")

    def toggle_sequencer(self):
        """Start or stop the sequencer"""
        if self.sequencer_running:
            self.stop_sequencer()
        else:
            self.start_sequencer()

    def start_sequencer(self):
        """Start the sequencer"""
        if self.sequencer_running:
            return

        self.sequencer_running = True
        self.start_btn.config(text="Stop Sequencer")
        self.sequencer_thread = threading.Thread(target=self.sequencer_loop)
        self.sequencer_thread.daemon = True
        self.sequencer_thread.start()

    def stop_sequencer(self):
        """Stop the sequencer"""
        self.sequencer_running = False
        self.start_btn.config(text="Start Sequencer")
        self.current_note_index = -1
        self.draw_note_indicators()

    def sequencer_loop(self):
        """Sequencer loop running in a separate thread"""
        while self.sequencer_running:
            for i in range(4):
                if not self.sequencer_running:
                    break

                self.current_note_index = i
                self.root.after(0, self.draw_note_indicators)

                # Play the note
                self.play_current_note()

                # Wait for the note duration (in samples converted to seconds)
                note_duration = self.sequencer_notes[i].get() / self.sample_rate
                time.sleep(note_duration)

    def play_current_note(self):
        """Play the current note"""
        # Generate a simple tone with the current envelope settings
        try:
            total_duration = self.decay_val.get() / 100.0 * 2.0
            t_audio = np.linspace(0, total_duration, int(total_duration * self.sample_rate), False)

            frequency = self.freq_var.get()
            carrier = np.sin(2 * np.pi * frequency * t_audio)

            A = self.attack_val.get()
            D = self.decay_val.get()

            envelope = np.zeros_like(t_audio)
            for i, t in enumerate(t_audio):
                t_scaled = t / total_duration * 100
                envelope[i] = self.compute_amplitude(t_scaled, A, D)

            audio_signal = carrier * envelope

            # Play audio in a non-blocking way
            sd.play(audio_signal, self.sample_rate)

        except Exception as e:
            print(f"Error playing note: {e}")

    def on_closing(self):
        """Stop audio when closing window"""
        self.stop_audio_playback()
        self.root.destroy()

    def save_shape_and_update(self):
        """Save shape and update display"""
        self.save_actual_shape()
        self.update_envelope()

    def set_reference_shape(self):
        self.ref_attack = self.attack_val.get()
        self.ref_decay = self.decay_val.get()
        self.update_envelope()

    def toggle_reference(self):
        self.show_reference = not self.show_reference
        self.toggle_ref_btn.config(text="Hide Reference" if self.show_reference else "Show Reference")
        self.update_envelope()

    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="Record Shape")
        else:
            # Start recording
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.actual_shape = []  # Reset recording buffer

        self.update_envelope()

    def save_actual_shape(self):
        """Save actual shape to saved shapes list"""
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

    def on_time_val_changed(self, *args):
        if self.clock_running and not self.updating_from_clock:
            self.stop_clock()

    def toggle_clock(self):
        if self.clock_running:
            self.stop_clock()
        else:
            self.start_clock()

    def start_clock(self):
        if not self.clock_running:
            self.clock_running = True
            self.play_btn.config(text="Stop")
            if self.time_val.get() >= self.sample_rate * 2:
                self.time_val.set(0)
            self.advance_clock()

    def stop_clock(self):
        if self.clock_running:
            self.clock_running = False
            self.play_btn.config(text="Play")
            # Stop recording when playback stops
            if self.recording:
                self.recording = False
                self.record_btn.config(text="Record Shape")

            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None

    def advance_clock(self):
        if not self.clock_running:
            return

        current_time = self.time_val.get()
        step = self.sample_rate / 100  # Move by 1% of sample rate each step
        new_time = current_time + step if current_time < self.sample_rate * 2 else 0

        self.updating_from_clock = True
        self.time_val.set(new_time)
        self.updating_from_clock = False

        # Record only if recording AND clock is running
        if self.recording and self.clock_running:
            try:
                # Convert sample time to percentage for recording
                t_percent = (new_time / (self.sample_rate * 2)) * 100
                amp = float(self.amplitude.get())
                self.actual_shape.append((t_percent, amp))
            except ValueError:
                pass

        # Update the envelope display
        self.update_envelope()

        # Schedule next update with a fixed interval
        self.after_id = self.root.after(100, self.advance_clock)

    def compute_amplitude(self, t, A, D):
        if A > 0 and t <= A:
            return t / A
        elif D > A and A <= t <= D:
            return 1.0 - (t - A) / (D - A)
        return 0.0

    def draw_envelope(self, A, D, t):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        padding = 20
        draw_width = width - 2 * padding
        draw_height = height - 2 * padding

        # Convert time from samples to percentage for display
        t_percent = (t / (self.sample_rate * 2)) * 100 if self.sample_rate * 2 > 0 else 0

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
            t_samples = self.time_val.get()
            A = self.attack_val.get()
            D = max(self.decay_val.get(), A)  # Ensure D >= A

            # Convert sample time to percentage for calculation
            t_percent = (t_samples / (self.sample_rate * 2)) * 100 if self.sample_rate * 2 > 0 else 0

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
            self.draw_envelope(A, D, t_samples)

        except (ValueError, ZeroDivisionError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = EnvelopeSimulator(root)
    root.mainloop()

