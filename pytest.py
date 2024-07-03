import tkinter as tk
import unittest
from unittest.mock import patch, MagicMock

import matplotlib
from birdCLEF import AudioAnalysisApp

matplotlib.use('Agg')


class TestAudioAnalysisApp(unittest.TestCase):
    def setUp(self):
        # Mock для tkinter
        self.root = MagicMock()
        tk.Tk = MagicMock(return_value=self.root)

    def test_create_widgets(self):
        app = AudioAnalysisApp()
        app.create_widgets()
        self.assertTrue(app.result_tabs.winfo_exists())
        self.assertTrue(app.result_tabs.winfo_children())

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_choose_audio_file(self, mock_askopenfilename):
        app = AudioAnalysisApp()
        app.choose_audio_file()
        self.assertEqual(app.audio_file_path, 'XC400498.ogg')
        self.assertTrue(mock_askopenfilename.called)

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_display_spectrogram(self, mock_spectrogram_tab):
        app = AudioAnalysisApp()
        app.start_analysis()
        self.assertTrue(mock_spectrogram_tab.winfo_children())

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_display_graphs(self, mock_graphs_tab):
        app = AudioAnalysisApp()
        app.start_analysis()
        self.assertTrue(mock_graphs_tab.winfo_children())

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_display_metrics(self, mock_metrix_tab):
        app = AudioAnalysisApp()
        app.start_analysis()
        self.assertTrue(mock_metrix_tab.winfo_children())

    @patch('birdCLEF.filedialog.askopenfilename', return_value='XC400498.ogg')
    def test_display_predictions(self, mock_predictions_tab):
        app = AudioAnalysisApp()
        app.start_analysis()
        self.assertTrue(mock_predictions_tab.winfo_children())


if __name__ == '__main__':
    unittest.main()
