import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from collections import Counter
import matplotlib.font_manager as fm
import mido
from mido import MidiFile

# 设置中文字体
plt.rcParams['font.family'] = ['Source Han Sans']
plt.rcParams['axes.unicode_minus'] = False


class MusicAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.notes = []
        self.valid_pitches = []
        self.current_file = ""

        # 定义音域范围
        self.vocal_ranges = {
            '男低音': ('E2', 'E4'),
            '男中音': ('G2', 'F4'),
            '男高音': ('B2', 'A4'),
            '女低音': ('F3', 'E5'),
            '女中音': ('A3', 'A5'),
            '女高音': ('C4', 'C6')
        }

    def initUI(self):
        self.setWindowTitle('音乐分析器 (支持MIDI/WAV/MP3)')
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 文件选择区域
        file_layout = QHBoxLayout()
        self.select_button = QPushButton('选择音乐文件', self)
        self.select_button.clicked.connect(self.select_file)
        self.file_label = QLabel('未选择文件')
        file_layout.addWidget(self.select_button)
        file_layout.addWidget(self.file_label)

        # 分析按钮
        self.analyze_button = QPushButton('开始分析', self)
        self.analyze_button.clicked.connect(self.analyze_file)
        self.analyze_button.setEnabled(False)

        # 状态标签
        self.status_label = QLabel('准备就绪')

        # 添加控件到主布局
        main_layout.addLayout(file_layout)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.status_label)

        # 创建选项卡
        self.tabs = QTabWidget()

        # 第一页：文本结果
        self.text_tab = QWidget()
        self.text_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.text_layout.addWidget(self.result_text)
        self.text_tab.setLayout(self.text_layout)

        # 第二页：音符频率表格
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout()
        self.note_table = QTableWidget()
        self.note_table.setColumnCount(13)
        headers = ["音符", "频率(%)"] + [f"+{i}半音" for i in range(1, 12)]
        self.note_table.setHorizontalHeaderLabels(headers)
        self.note_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.note_table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.note_table)
        self.table_tab.setLayout(self.table_layout)

        # 第三页：音域直方图
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        self.plot_tab.setLayout(self.plot_layout)

        # 添加选项卡
        self.tabs.addTab(self.text_tab, "分析结果")
        self.tabs.addTab(self.table_tab, "音符频率")
        self.tabs.addTab(self.plot_tab, "音域比较")

        main_layout.addWidget(self.tabs)

    def select_file(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                '选择音乐文件',
                '',
                '音频文件 (*.mp3 *.wav *.m4a *.ogg *.mid *.midi);;所有文件 (*)'
            )
            if file_name:
                self.current_file = file_name
                self.file_label.setText(f"已选择: {file_name}")
                self.analyze_button.setEnabled(True)
                self.status_label.setText('点击"开始分析"按钮进行分析')
                # 清空之前的结果
                self.result_text.clear()
                self.note_table.setRowCount(0)
                self.figure.clear()
                self.canvas.draw()
        except Exception as e:
            self.status_label.setText(f'文件选择出错: {str(e)}')

    def analyze_file(self):
        if not self.current_file:
            self.status_label.setText('请先选择文件')
            return

        file_ext = self.current_file.lower().split('.')[-1]
        try:
            self.status_label.setText('正在分析...')
            QApplication.processEvents()

            if file_ext in ['mid', 'midi']:
                self.analyze_midi(self.current_file)
            else:
                self.analyze_audio(self.current_file)

            self.status_label.setText('分析完成')
        except Exception as e:
            self.status_label.setText(f'分析出错: {str(e)}')
            self.result_text.setText(f"错误：{str(e)}\n\n{traceback.format_exc()}")

    def analyze_midi(self, file_path):
        """分析MIDI文件"""
        try:
            self.notes = []
            self.valid_pitches = []

            mid = MidiFile(file_path)
            tempo = 120  # 默认BPM

            # 查找设置速度的元消息
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = mido.tempo2bpm(msg.tempo)
                        break

            # 收集所有音符
            for track in mid.tracks:  # 修复：遍历所有轨道
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note = librosa.midi_to_note(msg.note)
                        self.notes.append(note)
                        self.valid_pitches.append(librosa.note_to_hz(note))

            # 准备分析结果
            result = "MIDI分析结果：\n\n"
            result += f"BPM: {tempo:.1f}\n"
            result += f"轨道数: {len(mid.tracks)}\n"
            result += f"总时长: {mid.length:.2f}秒\n"
            result += f"音符总数: {len(self.notes)}\n\n"

            # 添加音域和音符统计
            self.add_common_analysis_results(result)

            self.result_text.setText(result)
            self.update_note_table()
            self.update_pitch_histogram()

        except Exception as e:
            import traceback
            self.status_label.setText(f'MIDI分析出错: {str(e)}')
            self.result_text.setText(f"MIDI分析错误：{str(e)}\n\n{traceback.format_exc()}")


    def analyze_audio(self, file_path):
        """分析音频文件"""
        try:
            y, sr = librosa.load(file_path)

            # 分析BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # 分析音高
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            # 获取音符
            self.notes = []
            self.valid_pitches = []
            for i in range(pitches.shape[1]):
                pitch_slice = pitches[:, i]
                magnitude_slice = magnitudes[:, i]
                if np.any(magnitude_slice > 0):
                    max_magnitude_idx = magnitude_slice.argmax()
                    pitch = pitch_slice[max_magnitude_idx]
                    if pitch > 0:
                        note = librosa.hz_to_note(float(pitch))
                        self.notes.append(str(note))
                        self.valid_pitches.append(float(pitch))

            # 准备分析结果
            result = "音频分析结果：\n\n"
            result += f"BPM: {float(tempo):.1f}\n"
            result += f"采样率: {sr} Hz\n"
            result += f"时长: {len(y) / sr:.2f}秒\n\n"

            # 添加音域和音符统计
            self.add_common_analysis_results(result)

            self.result_text.setText(result)
            self.update_note_table()
            self.update_pitch_histogram()

        except Exception as e:
            raise Exception(f"音频分析错误: {str(e)}")

    def add_common_analysis_results(self, result):
        """添加通用的分析结果（音域和音符统计）"""
        # 音符统计
        if self.notes:
            note_counts = Counter(self.notes)
            total_notes = len(self.notes)

            result += "音符出现频率：\n"
            for note, count in note_counts.most_common():
                percentage = (count / total_notes) * 100
                result += f"{note}: {percentage:.1f}%\n"
        else:
            result += "未检测到音符\n"

        # 音域分析
        if self.valid_pitches:
            min_pitch = float(np.min(self.valid_pitches))
            max_pitch = float(np.max(self.valid_pitches))
            min_note = librosa.hz_to_note(min_pitch)
            max_note = librosa.hz_to_note(max_pitch)
            result += f"\n音域范围: {min_note} - {max_note}\n"

            # 判断音域范围
            result += "\n音域范围分析:\n"
            matched_ranges = []

            for range_name, (low_note, high_note) in self.vocal_ranges.items():
                low_hz = librosa.note_to_hz(low_note)
                high_hz = librosa.note_to_hz(high_note)

                # 检查音域是否完全包含在某个声部范围内
                if min_pitch >= low_hz and max_pitch <= high_hz:
                    matched_ranges.append(range_name)

            if matched_ranges:
                result += f"音域完全匹配: {', '.join(matched_ranges)}\n"
            else:
                # 如果没有完全匹配的，检查部分匹配
                for range_name, (low_note, high_note) in self.vocal_ranges.items():
                    low_hz = librosa.note_to_hz(low_note)
                    high_hz = librosa.note_to_hz(high_note)

                    # 检查音域是否有重叠
                    if (min_pitch <= high_hz and max_pitch >= low_hz):
                        overlap_min = max(min_pitch, low_hz)
                        overlap_max = min(max_pitch, high_hz)
                        overlap_percent = ((overlap_max - overlap_min) / (max_pitch - min_pitch)) * 100

                        if overlap_percent > 50:  # 如果重叠超过50%
                            result += f"音域部分匹配: {range_name} (重叠{overlap_percent:.1f}%)\n"

                # 如果没有匹配任何范围
                if not matched_ranges and "部分匹配" not in result:
                    result += "音域超出典型人声范围\n"

        # 添加音调分析
        if self.notes:
            note_roots = [note.replace('♯', '#')[0] for note in self.notes]
            root_counts = Counter(note_roots)
            most_common_root = root_counts.most_common(1)[0][0]
            result += f"\n可能的调式主音: {most_common_root}"

    def update_note_table(self):
        if not self.notes:
            return

        note_counts = Counter(self.notes)
        total_notes = len(self.notes)

        # 获取所有音符并按音高排序
        unique_notes = list(note_counts.keys())
        unique_notes.sort(key=lambda x: librosa.note_to_hz(x.replace('♯', '#').replace('♭', 'b')))

        # 准备表格数据
        table_data = []
        for note in unique_notes:
            count = note_counts[note]
            percentage = (count / total_notes) * 100
            row_data = [note, percentage]

            # 计算升半音对应的音符
            try:
                base_note = note.replace('♯', '#').replace('♭', 'b')
                for i in range(1, 12):  # +1到+11半音
                    hz = librosa.note_to_hz(base_note)
                    new_hz = hz * (2 ** (i / 12))
                    new_note = librosa.hz_to_note(new_hz)
                    # 统一转换为使用#表示升号
                    new_note = new_note.replace('♯', '#').replace('♭', 'b')
                    row_data.append(str(new_note))
            except:
                for i in range(1, 12):
                    row_data.append("N/A")

            table_data.append(row_data)

        # 设置表格行数
        self.note_table.setRowCount(len(table_data))

        # 填充表格
        for row, row_data in enumerate(table_data):
            note = row_data[0]

            # 创建音符单元格并设置颜色
            note_item = QTableWidgetItem(note)
            note_item.setFlags(note_item.flags() ^ Qt.ItemIsEditable)

            # 设置升降音颜色
            if '♯' in note or '#' in note:
                note_item.setBackground(QColor(255, 200, 200))  # 升号用浅红色
            elif '♭' in note or 'b' in note:
                note_item.setBackground(QColor(200, 200, 255))  # 降号用浅蓝色

            # 创建频率单元格
            freq_item = QTableWidgetItem(f"{row_data[1]:.1f}%")
            freq_item.setData(Qt.DisplayRole, row_data[1])  # 设置排序用的数值

            # 添加到表格
            self.note_table.setItem(row, 0, note_item)
            self.note_table.setItem(row, 1, freq_item)

            # 添加半音转换列
            for col in range(2, 13):  # 2-12列是+1到+11半音
                item = QTableWidgetItem(row_data[col])
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)

                # 为转换后的升降音也设置颜色
                converted_note = row_data[col]
                if '#' in converted_note:
                    item.setBackground(QColor(255, 200, 200))  # 升号用浅红色
                elif 'b' in converted_note:
                    item.setBackground(QColor(200, 200, 255))  # 降号用浅蓝色

                self.note_table.setItem(row, col, item)

        # 设置排序功能
        self.note_table.setSortingEnabled(True)
        self.note_table.sortByColumn(0, Qt.AscendingOrder)  # 默认按音符音高升序排列

    def update_pitch_histogram(self):
        if len(self.valid_pitches) == 0:
            return

        self.figure.clear()

        # 创建主图，调整底部空间以容纳声部标记
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.3)  # 增加底部空间

        # 将频率转换为音符
        note_names = [librosa.hz_to_note(p) for p in self.valid_pitches]
        unique_notes = sorted(list(set(note_names)), key=lambda x: librosa.note_to_hz(x))

        # 统计每个音符出现的次数
        note_counts = Counter(note_names)
        counts = [note_counts[note] for note in unique_notes]

        # 绘制柱状图（先绘制数据）
        bars = ax.bar(range(len(unique_notes)), counts, color='skyblue', edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(unique_notes)))
        ax.set_xticklabels(unique_notes)

        # 定义不同声部的颜色（带透明度）
        range_colors = {
            '男低音': (0.68, 0.85, 0.90, 0.5),  # lightblue 带50%透明度
            '男中音': (0.0, 0.0, 1.0, 0.5),  # blue 带50%透明度
            '男高音': (0.0, 0.0, 0.55, 0.5),  # darkblue 带50%透明度
            '女低音': (1.0, 0.75, 0.8, 0.5),  # pink 带50%透明度
            '女中音': (1.0, 0.0, 0.0, 0.5),  # red 带50%透明度
            '女高音': (0.55, 0.0, 0.0, 0.5)  # darkred 带50%透明度
        }

        # 准备范围标记数据
        range_data = []
        for range_name, (low_note, high_note) in self.vocal_ranges.items():
            try:
                low_hz = librosa.note_to_hz(low_note)
                high_hz = librosa.note_to_hz(high_note)

                # 找到范围内的音符索引
                start_idx = None
                end_idx = None
                for i, note in enumerate(unique_notes):
                    note_hz = librosa.note_to_hz(note)
                    if start_idx is None and note_hz >= low_hz:
                        start_idx = i
                    if note_hz <= high_hz:
                        end_idx = i
                    elif note_hz > high_hz:
                        break

                if start_idx is not None and end_idx is not None:
                    range_data.append({
                        'name': range_name,
                        'start': start_idx,
                        'end': end_idx,
                        'color': range_colors[range_name]
                    })
            except:
                continue

        # 在图表底部创建声部范围标记（半透明）
        for i, range_info in enumerate(range_data):
            # 计算每个声部标记的y位置（从0.02开始，每个声部间隔0.08）
            y_pos = 0.02 + (i * 0.08)

            # 绘制半透明范围标记（不遮挡数据）
            ax.broken_barh(
                [(range_info['start'], range_info['end'] - range_info['start'] + 1)],
                (y_pos, 0.06),  # 高度设为0.06
                facecolors=range_info['color'],  # 使用带透明度的颜色
                edgecolor=(0.2, 0.2, 0.2, 0.7),  # 半透明边框
                linewidth=0.8,
                transform=ax.get_xaxis_transform(),
                label=range_info['name']
            )

        # 设置轴标签
        ax.set_xlabel('音符')
        ax.set_ylabel('出现次数')
        ax.set_title('音域分布（按音符）')

        # 添加图例（放在图表右侧）
        legend_handles = []
        for range_info in range_data:
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1,
                fc=range_info['color'],
                label=range_info['name'],
                alpha=0.5  # 图例也保持半透明
            ))
        ax.legend(handles=legend_handles,
                  loc='upper left',
                  bbox_to_anchor=(1.02, 1),
                  framealpha=0.7)  # 图例背景半透明

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 调整布局防止标签重叠
        self.figure.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = MusicAnalyzer()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
