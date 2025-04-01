import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from collections import Counter
import matplotlib.font_manager as fm
import mido
from mido import MidiFile
import traceback # Moved import to top

# --- Font Setup ---
# Try to set Chinese fonts with fallbacks
try:
    # Prioritize Source Han Sans, fallback to common Chinese fonts
    plt.rcParams['font.family'] = ['Source Han Sans CN', 'SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("Using font:", plt.rcParams['font.family'])
except Exception as e:
    print(f"Warning: Could not set preferred Chinese font. Using default. Error: {e}")
    # Use a generic sans-serif font if others fail
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# --- Analysis Worker Thread ---
class AnalysisWorker(QThread):
    """Runs the analysis in a separate thread to avoid freezing the GUI."""
    # Signals: finished(results_dict), error(error_msg, traceback_str), progress(message)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.file_ext = file_path.lower().split('.')[-1]

    def run(self):
        """Performs the analysis."""
        try:
            self.progress.emit('正在分析...')
            QApplication.processEvents() # Allow UI updates during setup

            results = {} # Dictionary to hold all analysis results
            if self.file_ext in ['mid', 'midi']:
                results = self._analyze_midi()
            elif self.file_ext in ['wav', 'mp3', 'm4a', 'ogg', 'flac']: # Added flac
                results = self._analyze_audio()
            else:
                raise ValueError(f"不支持的文件格式: .{self.file_ext}")

            self.progress.emit('分析完成')
            self.finished.emit(results)

        except Exception as e:
            tb_str = traceback.format_exc()
            self.error.emit(f'分析过程中出现错误: {str(e)}', tb_str)

    def _analyze_midi(self):
        """Analyzes a MIDI file and returns results."""
        notes = []
        valid_pitches_hz = []
        tempo = 120.0  # Default BPM
        duration = 0.0
        num_tracks = 0

        mid = MidiFile(self.file_path)
        num_tracks = len(mid.tracks)
        duration = mid.length

        # Find tempo (use the first encountered tempo)
        tempo_found = False
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
                    tempo_found = True
                    break
            if tempo_found:
                break

        # Collect notes
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time # Accumulate time for potential duration calculation later
                if msg.type == 'note_on' and msg.velocity > 0:
                    try:
                        # Use librosa for consistent note naming (e.g., C#4)
                        note_name = librosa.midi_to_note(msg.note, unicode=False) # Use ASCII sharps (#)
                        note_hz = librosa.midi_to_hz(msg.note)
                        notes.append(note_name)
                        valid_pitches_hz.append(note_hz)
                    except Exception as note_e:
                        print(f"Warning: Could not process MIDI note {msg.note}. Error: {note_e}")
                        continue # Skip problematic notes

        return {
            'type': 'midi',
            'notes': notes,
            'valid_pitches_hz': valid_pitches_hz,
            'tempo': tempo,
            'duration': duration,
            'num_tracks': num_tracks,
            'sample_rate': None # Not applicable for MIDI
        }

    def _analyze_audio(self):
        """Analyzes an audio file and returns results."""
        self.progress.emit('正在加载音频...')
        QApplication.processEvents()
        y, sr = librosa.load(self.file_path, sr=None) # Load with original sample rate
        duration = librosa.get_duration(y=y, sr=sr)

        self.progress.emit('正在分析BPM...')
        QApplication.processEvents()
        # Use a more robust tempo estimation if available, or stick with beat_track
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr) # Remove units='bpm'

        self.progress.emit('正在分析音高...')
        QApplication.processEvents()
        # --- Pitch Tracking ---
        # Note: piptrack is basic. For better results, consider CREPE or YIN algorithms,
        # but they add complexity and dependencies.
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        notes = []
        valid_pitches_hz = []
        # Select pitches with highest magnitude in each frame
        # This is a simplification and may not be accurate for complex audio
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch_hz = pitches[index, t]
            if pitch_hz > 0: # Filter out 0 Hz estimates
                 try:
                    # Use librosa for consistent note naming (e.g., C#4)
                    note_name = librosa.hz_to_note(pitch_hz, unicode=False) # Use ASCII sharps (#)
                    notes.append(note_name)
                    valid_pitches_hz.append(pitch_hz)
                 except Exception as note_e:
                    print(f"Warning: Could not process pitch {pitch_hz}Hz. Error: {note_e}")
                    continue # Skip problematic pitches

        return {
            'type': 'audio',
            'notes': notes,
            'valid_pitches_hz': valid_pitches_hz,
            'tempo': float(tempo), # Ensure tempo is float
            'duration': duration,
            'num_tracks': None, # Not applicable for single audio file
            'sample_rate': sr
        }


# --- Main Application Window ---
class MusicAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.notes = [] # Store detected note names (e.g., 'C#4')
        self.valid_pitches_hz = [] # Store detected pitches in Hz
        self.current_file = ""
        self.analysis_worker = None # Placeholder for the worker thread

        # Vocal ranges using standard note names (ASCII sharps)
        self.vocal_ranges = {
            '男低音': ('E2', 'E4'),
            '男中音': ('G2', 'F4'),
            '男高音': ('B2', 'A4'),
            '女低音': ('F3', 'E5'),
            '女中音': ('A3', 'A5'),
            '女高音': ('C4', 'C6')
        }
        # Pre-calculate Hz ranges for faster comparison
        self.vocal_ranges_hz = {
            name: (librosa.note_to_hz(low), librosa.note_to_hz(high))
            for name, (low, high) in self.vocal_ranges.items()
        }

        self.initUI()

    def initUI(self):
        self.setWindowTitle('音乐分析器 (支持MIDI/WAV/MP3/M4A/OGG/FLAC)')
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Controls ---
        control_layout = QHBoxLayout()
        self.select_button = QPushButton('选择音乐文件', self)
        self.select_button.clicked.connect(self.select_file)
        self.file_label = QLabel('未选择文件')
        self.file_label.setStyleSheet("QLabel { padding-left: 5px; }") # Add padding
        self.analyze_button = QPushButton('开始分析', self)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)

        control_layout.addWidget(self.select_button)
        control_layout.addWidget(self.file_label, 1) # Allow label to stretch
        control_layout.addWidget(self.analyze_button)
        main_layout.addLayout(control_layout)

        # --- Status Bar ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel('准备就绪')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0) # Indeterminate progress
        self.progress_bar.setVisible(False) # Hide initially
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)


        # --- Tabs ---
        self.tabs = QTabWidget()

        # Tab 1: Text Results
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFontFamily("monospace") # Monospace font for better alignment
        text_layout.addWidget(self.result_text)

        # Tab 2: Note Frequency Table
        self.table_tab = QWidget()
        table_layout = QVBoxLayout(self.table_tab)
        self.note_table = QTableWidget()
        self.note_table.setColumnCount(13) # Note, Freq(%), +1 to +11 semitones
        headers = ["音符", "频率(%)"] + [f"+{i}半音" for i in range(1, 12)]
        self.note_table.setHorizontalHeaderLabels(headers)
        self.note_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.note_table.horizontalHeader().setStretchLastSection(False) # Don't stretch last column
        self.note_table.setSortingEnabled(True) # Enable sorting
        self.note_table.setAlternatingRowColors(True) # Improve readability
        table_layout.addWidget(self.note_table)

        # Tab 3: Pitch Histogram Plot
        self.plot_tab = QWidget()
        plot_layout = QVBoxLayout(self.plot_tab)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Add tabs
        self.tabs.addTab(self.text_tab, "分析结果")
        self.tabs.addTab(self.table_tab, "音符频率")
        self.tabs.addTab(self.plot_tab, "音域比较")

        main_layout.addWidget(self.tabs)

    def select_file(self):
        """Opens a dialog to select a music file."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                '选择音乐文件',
                '',
                '音频和MIDI文件 (*.mp3 *.wav *.m4a *.ogg *.flac *.mid *.midi);;所有文件 (*)'
            )
            if file_name:
                self.current_file = file_name
                self.file_label.setText(f"已选择: {file_name}")
                self.analyze_button.setEnabled(True)
                self.status_label.setText('准备分析')
                # Clear previous results immediately
                self.clear_results()
        except Exception as e:
            self.status_label.setText(f'文件选择出错: {str(e)}')
            self.analyze_button.setEnabled(False)

    def clear_results(self):
        """Clears all previous analysis results from the UI."""
        self.result_text.clear()
        self.note_table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw()
        self.notes = []
        self.valid_pitches_hz = []

    def start_analysis(self):
        """Starts the analysis process in a separate thread."""
        if not self.current_file or self.analysis_worker is not None:
            return # Prevent starting if no file or analysis already running

        self.clear_results() # Clear results before starting new analysis
        self.analyze_button.setEnabled(False)
        self.select_button.setEnabled(False) # Disable file selection during analysis
        self.status_label.setText('正在初始化分析...')
        self.progress_bar.setVisible(True)

        # Create and start the worker thread
        self.analysis_worker = AnalysisWorker(self.current_file)
        self.analysis_worker.finished.connect(self.handle_analysis_success)
        self.analysis_worker.error.connect(self.handle_analysis_error)
        self.analysis_worker.progress.connect(self.update_status)
        # Clean up thread when finished
        self.analysis_worker.finished.connect(self.analysis_complete)
        self.analysis_worker.error.connect(self.analysis_complete)
        self.analysis_worker.start()

    def update_status(self, message):
        """Updates the status label."""
        self.status_label.setText(message)

    def handle_analysis_success(self, results):
        """Handles successful analysis results from the worker thread."""
        self.status_label.setText('分析完成，正在处理结果...')
        QApplication.processEvents()

        # Store results
        self.notes = results.get('notes', [])
        self.valid_pitches_hz = results.get('valid_pitches_hz', [])

        # --- Build Text Report ---
        report = []
        analysis_type = results.get('type', '未知')
        report.append(f"{analysis_type.upper()} 分析结果:\n")
        report.append(f"文件: {self.current_file}\n")

        if results.get('tempo') is not None:
            report.append(f"估算 BPM: {results['tempo']:.1f}")
        if results.get('duration') is not None:
            report.append(f"时长: {results['duration']:.2f} 秒")
        if results.get('sample_rate') is not None:
            report.append(f"采样率: {results['sample_rate']} Hz")
        if results.get('num_tracks') is not None:
            report.append(f"轨道数: {results['num_tracks']}")

        report.append(f"检测到的音符事件数: {len(self.notes)}") # Use len(notes) as proxy

        # --- Add Common Analysis (Range, Key) ---
        common_analysis_text = self.generate_common_analysis_report()
        if common_analysis_text:
            report.append("\n--- 综合分析 ---\n" + common_analysis_text)

        self.result_text.setText("\n".join(report))

        # --- Update Table and Plot ---
        self.update_note_table()
        self.update_pitch_histogram()

        self.status_label.setText('分析完成')

    def handle_analysis_error(self, error_msg, tb_str):
        """Handles errors reported by the worker thread."""
        self.status_label.setText(f'分析失败: {error_msg}')
        self.result_text.setText(f"分析过程中发生错误:\n\n{error_msg}\n\n详细信息:\n{tb_str}")
        # Keep plot/table clear or show an error message there? Clear is simpler.
        self.figure.clear()
        self.canvas.draw()
        self.note_table.setRowCount(0)


    def analysis_complete(self):
        """Called when the worker thread finishes (success or error)."""
        self.analyze_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.analysis_worker:
            # Ensure thread resources are released (though Python's GC usually handles it)
             self.analysis_worker.quit()
             self.analysis_worker.wait() # Wait for thread to fully terminate
        self.analysis_worker = None # Reset worker


    def generate_common_analysis_report(self):
        """Generates text report for note frequency, range, and key."""
        report_parts = []

        # --- Note Frequency ---
        if self.notes:
            note_counts = Counter(self.notes)
            total_notes = len(self.notes)
            report_parts.append("主要音符频率:")
            # Show top 5 or up to 10 notes
            for note, count in note_counts.most_common(10):
                percentage = (count / total_notes) * 100
                report_parts.append(f"  {note:<4}: {percentage:>5.1f}% ({count}次)")
            if len(note_counts) > 10:
                report_parts.append("  ...")
        else:
            report_parts.append("未检测到明确的音符事件。")

        # --- Pitch Range Analysis ---
        if self.valid_pitches_hz:
            min_pitch_hz = float(np.min(self.valid_pitches_hz))
            max_pitch_hz = float(np.max(self.valid_pitches_hz))
            min_note = librosa.hz_to_note(min_pitch_hz, unicode=False)
            max_note = librosa.hz_to_note(max_pitch_hz, unicode=False)
            report_parts.append(f"\n音域范围 (Hz): {min_pitch_hz:.2f} Hz - {max_pitch_hz:.2f} Hz")
            report_parts.append(f"音域范围 (音符): {min_note} - {max_note}")

            # --- Vocal Range Matching ---
            report_parts.append("\n人声声部匹配:")
            matched_ranges = []
            partial_matches = []

            for range_name, (low_hz, high_hz) in self.vocal_ranges_hz.items():
                # Check for full containment
                if min_pitch_hz >= low_hz and max_pitch_hz <= high_hz:
                    matched_ranges.append(range_name)
                else:
                    # Check for significant overlap
                    overlap_min = max(min_pitch_hz, low_hz)
                    overlap_max = min(max_pitch_hz, high_hz)
                    if overlap_max > overlap_min: # Ensure there is *some* overlap
                        detected_range_width = max_pitch_hz - min_pitch_hz
                        if detected_range_width > 1e-6: # Avoid division by zero/very small
                            overlap_percentage = ((overlap_max - overlap_min) / detected_range_width) * 100
                            # Define a threshold for "significant" overlap, e.g., 50%
                            if overlap_percentage >= 50.0:
                                partial_matches.append(f"{range_name} (重叠 {overlap_percentage:.1f}%)")
                        # Handle case where detected range is very narrow but overlaps
                        elif min_pitch_hz >= low_hz and max_pitch_hz <= high_hz:
                             partial_matches.append(f"{range_name} (窄音域内)")


            if matched_ranges:
                report_parts.append(f"  完全匹配: {', '.join(matched_ranges)}")
            if partial_matches:
                report_parts.append(f"  部分匹配: {', '.join(partial_matches)}")
            if not matched_ranges and not partial_matches:
                report_parts.append("  未显著匹配标准人声声部。")

        else:
             report_parts.append("\n无法进行音域分析 (未检测到音高)。")


        # --- Basic Key Estimation ---
        if self.notes:
            # Extract root note (handle sharps '#' and flats 'b')
            # Assumes notes like 'C#4', 'Db5' etc.
            note_roots = [note[:-1].replace('#', '').replace('b', '') for note in self.notes if len(note) > 1]
            if note_roots:
                root_counts = Counter(note_roots)
                most_common_root = root_counts.most_common(1)[0][0]
                report_parts.append(f"\n可能的调式主音 (基于频率): {most_common_root}")
            else:
                report_parts.append("\n无法估算主音 (音符格式不明确)。")


        return "\n".join(report_parts)


    def update_note_table(self):
        """Populates the note frequency table."""
        if not self.notes:
            self.note_table.setRowCount(0)
            return

        note_counts = Counter(self.notes)
        total_notes = len(self.notes)

        # Get unique notes and sort them by pitch (using Hz for reliable sorting)
        # Ensure consistent sharp/flat handling for sorting
        unique_notes = sorted(
            list(note_counts.keys()),
            key=lambda n: librosa.note_to_hz(n) if n else 0 # Handle potential None/empty notes
        )

        self.note_table.setRowCount(len(unique_notes))
        self.note_table.setSortingEnabled(False) # Disable sorting during population

        for row, note in enumerate(unique_notes):
            if not note: continue # Skip empty entries if any

            count = note_counts[note]
            percentage = (count / total_notes) * 100

            # Column 0: Note Name
            note_item = QTableWidgetItem(note)
            note_item.setFlags(note_item.flags() ^ Qt.ItemIsEditable)
            # Color coding for sharps/flats
            if '#' in note:
                note_item.setBackground(QColor(255, 220, 220)) # Light red for sharp
            elif 'b' in note:
                note_item.setBackground(QColor(220, 220, 255)) # Light blue for flat
            self.note_table.setItem(row, 0, note_item)

            # Column 1: Frequency (%) - Store float for sorting
            freq_item = QTableWidgetItem(f"{percentage:.1f}%")
            freq_item.setData(Qt.DisplayRole, f"{percentage:.1f}%") # Display text
            freq_item.setData(Qt.UserRole, percentage) # Store float data for sorting
            freq_item.setFlags(freq_item.flags() ^ Qt.ItemIsEditable)
            freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.note_table.setItem(row, 1, freq_item)

            # Columns 2-12: Semitone Transpositions
            try:
                base_hz = librosa.note_to_hz(note)
                for i in range(1, 12): # +1 to +11 semitones
                    transposed_hz = base_hz * (2**(i/12.0))
                    transposed_note = librosa.hz_to_note(transposed_hz, unicode=False) # Use ASCII sharps
                    item = QTableWidgetItem(transposed_note)
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    # Apply color coding to transposed notes
                    if '#' in transposed_note:
                        item.setBackground(QColor(255, 230, 230))
                    elif 'b' in transposed_note:
                        item.setBackground(QColor(230, 230, 255))
                    self.note_table.setItem(row, i + 1, item) # Offset by 1 (col 2 is +1)
            except Exception as e:
                print(f"Warning: Could not transpose note {note}. Error: {e}")
                for i in range(1, 12):
                    item = QTableWidgetItem("N/A")
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    self.note_table.setItem(row, i + 1, item)

        self.note_table.setSortingEnabled(True)
        # Optional: Sort by note name initially
        # self.note_table.sortByColumn(0, Qt.AscendingOrder)
        # Or sort by frequency descending
        self.note_table.sortByColumn(1, Qt.DescendingOrder)


    def update_pitch_histogram(self):
        """Updates the pitch histogram plot."""
        self.figure.clear() # Clear previous plot

        if not self.notes:
            # Optionally display a message on the canvas
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '无音符数据可供绘图', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        # Use self.notes which should be populated correctly now
        note_counts = Counter(self.notes)

        # Sort notes by pitch for the x-axis
        sorted_unique_notes = sorted(
            list(note_counts.keys()),
            key=lambda n: librosa.note_to_hz(n) if n else 0
        )
        counts = [note_counts[note] for note in sorted_unique_notes]

        # --- Plotting ---
        ax = self.figure.add_subplot(111)
        # Adjust bottom margin to make space for range labels
        self.figure.subplots_adjust(bottom=0.3, right=0.80) # Adjust right for legend

        # Bar chart for note counts
        indices = np.arange(len(sorted_unique_notes))
        bars = ax.bar(indices, counts, color='skyblue', edgecolor='black', alpha=0.8, label='出现次数')
        ax.set_xticks(indices)
        ax.set_xticklabels(sorted_unique_notes, rotation=45, ha='right', fontsize=8) # Smaller font if many notes
        ax.set_xlabel('音符')
        ax.set_ylabel('出现次数')
        ax.set_title('音符频率分布与人声声部比较')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # --- Vocal Range Overlays ---
        # Define colors with alpha for overlays
        range_colors = {
            '男低音': (0.68, 0.85, 0.90, 0.3), '男中音': (0.0, 0.0, 1.0, 0.3), '男高音': (0.0, 0.0, 0.55, 0.3),
            '女低音': (1.0, 0.75, 0.8, 0.3), '女中音': (1.0, 0.0, 0.0, 0.3), '女高音': (0.55, 0.0, 0.0, 0.3)
        }
        legend_handles = [] # For custom legend

        # Calculate y-positions for range labels dynamically
        num_ranges = len(self.vocal_ranges_hz)
        y_step = 0.06 # Height of each range bar in relative coords
        y_gap = 0.01  # Gap between range bars
        total_range_height = num_ranges * y_step + (num_ranges - 1) * y_gap
        y_start = 0.02 # Starting y-position from bottom

        # Map note names to indices for plotting ranges
        note_to_index = {note: i for i, note in enumerate(sorted_unique_notes)}

        for i, (range_name, (low_hz, high_hz)) in enumerate(self.vocal_ranges_hz.items()):
            # Find the indices corresponding to the range
            start_idx = -1
            end_idx = -1
            for note, idx in note_to_index.items():
                note_hz = librosa.note_to_hz(note)
                if start_idx == -1 and note_hz >= low_hz:
                    start_idx = idx
                if note_hz <= high_hz:
                    end_idx = idx
                # Optimization: if current note is already above range, break early
                # (assumes sorted_unique_notes)
                # if note_hz > high_hz and start_idx != -1:
                #      break # No need to check further notes for this range

            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                # Calculate position and width for broken_barh
                # Add 0.5 to center the bar visually over the indices
                x_start = start_idx - 0.5
                x_width = (end_idx - start_idx) + 1

                # Calculate y position for this range bar
                y_pos = y_start + i * (y_step + y_gap)

                # Draw the semi-transparent range bar using axis coordinates for y
                ax.broken_barh(
                    [(x_start, x_width)],
                    (y_pos, y_step), # y position and height in relative coords
                    facecolors=range_colors[range_name],
                    edgecolor=(0.2, 0.2, 0.2, 0.5),
                    linewidth=0.8,
                    transform=ax.get_xaxis_transform() # Key: X=data coords, Y=axis coords
                )
                # Add range name text slightly above the bar
                # ax.text(x_start + x_width / 2, y_pos + y_step / 2, range_name,
                #         ha='center', va='center', fontsize=7, color='black',
                #         transform=ax.get_xaxis_transform()) # Also use transform

            # Create proxy artist for the legend
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=range_colors[range_name], label=range_name))


        # Add legend outside the plot area
        ax.legend(handles=legend_handles, title="声部范围",
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # Final adjustments
        # self.figure.tight_layout(rect=[0, 0.05, 0.85, 0.95]) # Adjust rect to fit legend/labels
        ax.margins(x=0.02) # Add small margin to x-axis

        self.canvas.draw()

    def closeEvent(self, event):
        """Ensure worker thread is stopped on close."""
        if self.analysis_worker and self.analysis_worker.isRunning():
            print("Attempting to stop analysis worker...")
            # You might want to signal the worker to stop gracefully if possible
            # For now, we just quit and wait briefly
            self.analysis_worker.quit()
            if not self.analysis_worker.wait(1000): # Wait max 1 sec
                 print("Warning: Analysis worker did not stop gracefully.")
                 # self.analysis_worker.terminate() # Force terminate if needed (use with caution)
        event.accept()


def main():
    app = QApplication(sys.argv)
    # Optional: Apply a style for better look and feel
    try:
        app.setStyle('Fusion')
    except Exception as e:
        print(f"Could not set Fusion style: {e}")

    ex = MusicAnalyzer()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
