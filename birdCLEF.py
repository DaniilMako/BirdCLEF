import tkinter as tk
from tkinter import filedialog, ttk
from warnings import filterwarnings

import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
# import torch
from torch import device, cuda, no_grad, topk, load, flatten
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from timm import create_model
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

filterwarnings("ignore")


class AudioAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.result_tabs = ttk.Notebook(self)
        self.spectrogram_tab = tk.Frame(self.result_tabs)
        self.graphs_tab = tk.Frame(self.result_tabs)
        self.metrics_tab = tk.Frame(self.result_tabs)
        self.predictions_tab = tk.Frame(self.result_tabs)
        self.title("Audio Analysis App")
        self.geometry("1600x900")

        self.audio_file_path = None
        # self.audio_file_path = 'asserts/audio/XC400498.ogg'
        self.mode = tk.StringVar(value="С тишиной")
        self.experiment = tk.StringVar(value="Mel")
        self.model = tk.StringVar(value="ResNet182")
        self.model_file = None

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.birds = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1',
                      'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1',
                      'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1',
                      'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1',
                      'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1',
                      'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2',
                      'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov',
                      'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1',
                      'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan',
                      'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1',
                      'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1',
                      'ingori1', 'inpher1', 'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2',
                      'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1',
                      'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2',
                      'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1',
                      'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3',
                      'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3',
                      'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar',
                      'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1',
                      'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1',
                      'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2',
                      'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun',
                      'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
        self.transform = Compose([
            Resize((200, 800)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.25])
        ])

        self.create_widgets()

    def create_widgets(self):
        # Frame for file selection
        file_frame = tk.Frame(self)
        file_frame.pack(pady=5)

        tk.Label(file_frame, text="Выберите аудио:").pack(side=tk.LEFT)
        tk.Button(file_frame, text="Обзор", command=self.choose_audio_file).pack(side=tk.LEFT)

        # Frame for AI model
        model_frame = tk.Frame(self)
        model_frame.pack(pady=5)

        tk.Label(model_frame, text="Модель:").pack(side=tk.LEFT)
        tk.Radiobutton(model_frame, text="ResNet182", variable=self.model, value="ResNet182").pack(side=tk.LEFT)
        tk.Radiobutton(model_frame, text="EfficientNet_b0", variable=self.model,
                       value="EfficientNet_b0").pack(side=tk.LEFT)

        # Frame for mode and experiment selection
        mode_frame = tk.Frame(self)
        mode_frame.pack(pady=5)

        tk.Label(mode_frame, text="Режим:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="С тишиной", variable=self.mode, value="С тишиной").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Без тишины", variable=self.mode, value="Без тишины").pack(side=tk.LEFT)

        experiment_frame = tk.Frame(self)
        experiment_frame.pack(pady=5)

        tk.Label(experiment_frame, text="Эксперимент:").pack(side=tk.LEFT)
        tk.OptionMenu(experiment_frame, self.experiment, "Mel", "Spectral Contrast",
                      "Constant Q Transform", "Short Time Fourier Transform", "Chroma Contrast", "Label Smoothing",
                      "Augmentations").pack(side=tk.LEFT)

        # Начало анализа
        tk.Button(self, text="Начать анализ", command=self.start_analysis).pack()

        # Таблицы с результатами
        self.result_tabs.pack(expand=True, fill=tk.BOTH)

        self.result_tabs.add(self.spectrogram_tab, text="Спектрограмма")

        self.result_tabs.add(self.graphs_tab, text="Графики")

        self.result_tabs.add(self.metrics_tab, text="Метрики")

        self.result_tabs.add(self.predictions_tab, text="Предсказания")

    def start_analysis(self):
        if self.audio_file_path:
            print("Выбранная модель:", self.model.get())
            print("Выбранный режим:", self.mode.get())
            print("Выбранный эксперимент:", self.experiment.get())
            self.display_spectrogram()
            self.display_graphs()
            self.display_metrics()
            self.display_predictions()
            self.true_to_false()

    def choose_audio_file(self):
        self.audio_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.ogg;*.mp3;*.wav")])
        if self.audio_file_path:
            print("Выбран аудиофайл:", self.audio_file_path)

    def display_spectrogram(self):
        # Load audio file
        y, sr = librosa.load(self.audio_file_path)
        if self.mode.get() == "Без тишины":
            top_db_c = 10
            non_silent_intervals = librosa.effects.split(y, top_db=top_db_c)
            y = np.concatenate([y[start:end] for start, end in non_silent_intervals])

        if self.experiment.get() == "Mel":
            # Create spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Plot spectrogram
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel Spectrogram')

            # Clear previous spectrogram if exists
            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Spectral Contrast":
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            plt.figure(figsize=(8, 4))
            freqs = librosa.cqt_frequencies(spectral_contrast.shape[0], fmin=librosa.note_to_hz('C1'))
            librosa.display.specshow(spectral_contrast, x_axis='time', y_axis='linear', sr=sr,
                                     fmin=freqs.min(), fmax=freqs.max())
            plt.colorbar()
            # plt.title('Spectral Contrast')

            # Clear previous spectrogram if exists
            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Constant Q Transform":
            cqt_spectrogram = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr), ref=np.max)
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(cqt_spectrogram, sr=sr, x_axis='time', y_axis='cqt_note')
            plt.colorbar(format='%+2.0f дБ')
            # plt.title('Constant Q Spectrogram')

            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Short Time Fourier Transform":
            stft_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(stft_spectrogram, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f дБ')
            # plt.title('Short-Time Fourier Transform Spectrogram')

            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Chroma Contrast":
            chroma_contrast = librosa.feature.chroma_cens(y=y, sr=sr)
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(chroma_contrast, x_axis='time', y_axis='chroma')
            plt.colorbar()
            # plt.title('Chroma Contrast')

            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Label Smoothing":
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel Spectrogram')

            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        elif self.experiment.get() == "Augmentations":
            noise = np.random.randn(len(y))
            augmented_data = y + 0.005 * noise
            augmented_data = np.clip(augmented_data, -1, 1)
            y = augmented_data

            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            plt.figure(figsize=(8, 4))
            librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel with Augmentations Spectrogram')

            for widget in self.spectrogram_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.spectrogram_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # return self.experiment.get()
        return True

    def display_graphs(self):
        # Clear previous graphs if exists
        for widget in self.graphs_tab.winfo_children():
            widget.destroy()

        dir_path = "asserts\\graphs\\"

        if self.model.get() == "ResNet182":
            if self.mode.get() == "С тишиной":
                if self.experiment.get() == "Mel":
                    graph_image_path = f"{dir_path}graph_resnet_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    graph_image_path = f"{dir_path}graph_resnet_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    graph_image_path = f"{dir_path}graph_resnet_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    graph_image_path = f"{dir_path}graph_resnet_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    graph_image_path = f"{dir_path}graph_resnet_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    graph_image_path = f"{dir_path}graph_resnet_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    graph_image_path = f"{dir_path}graph_resnet_aug.jpg"
            else:
                if self.experiment.get() == "Mel":
                    graph_image_path = f"{dir_path}graph_resnet_rs_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    graph_image_path = f"{dir_path}graph_resnet_rs_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    graph_image_path = f"{dir_path}graph_resnet_rs_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    graph_image_path = f"{dir_path}graph_resnet_rs_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    graph_image_path = f"{dir_path}graph_resnet_rs_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    graph_image_path = f"{dir_path}graph_resnet_rs_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    graph_image_path = f"{dir_path}graph_resnet_rs_aug.jpg"
        else:
            if self.mode.get() == "С тишиной":
                if self.experiment.get() == "Mel":
                    graph_image_path = f"{dir_path}graph_efnet_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    graph_image_path = f"{dir_path}graph_efnet_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    graph_image_path = f"{dir_path}graph_efnet_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    graph_image_path = f"{dir_path}graph_efnet_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    graph_image_path = f"{dir_path}graph_efnet_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    graph_image_path = f"{dir_path}graph_efnet_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    graph_image_path = f"{dir_path}graph_efnet_aug.jpg"
            else:
                if self.experiment.get() == "Mel":
                    graph_image_path = f"{dir_path}graph_efnet_rs_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    graph_image_path = f"{dir_path}graph_efnet_rs_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    graph_image_path = f"{dir_path}graph_efnet_rs_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    graph_image_path = f"{dir_path}graph_efnet_rs_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    graph_image_path = f"{dir_path}graph_efnet_rs_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    graph_image_path = f"{dir_path}graph_efnet_rs_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    graph_image_path = f"{dir_path}graph_efnet_rs_aug.jpg"

        graph_image = Image.open(graph_image_path)
        graph_photo = ImageTk.PhotoImage(graph_image)
        graph_label = tk.Label(self.graphs_tab, image=graph_photo)
        graph_label.image = graph_photo  # Keep a reference to prevent garbage collection
        graph_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def display_metrics(self):
        # Clear previous metrics if exists
        for widget in self.metrics_tab.winfo_children():
            widget.destroy()

        dir_path = "asserts\\metrics\\"
        metric_image_path = f"{dir_path}metric_resnet_mel.png"

        if self.model.get() == "ResNet182":
            if self.mode.get() == "С тишиной":
                if self.experiment.get() == "Mel":
                    metric_image_path = f"{dir_path}metric_resnet_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    metric_image_path = f"{dir_path}metric_resnet_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    metric_image_path = f"{dir_path}metric_resnet_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    metric_image_path = f"{dir_path}metric_resnet_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    metric_image_path = f"{dir_path}metric_resnet_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    metric_image_path = f"{dir_path}metric_resnet_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    metric_image_path = f"{dir_path}metric_resnet_aug.jpg"
            else:
                if self.experiment.get() == "Mel":
                    metric_image_path = f"{dir_path}metric_resnet_rs_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    metric_image_path = f"{dir_path}metric_resnet_rs_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    metric_image_path = f"{dir_path}metric_resnet_rs_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    metric_image_path = f"{dir_path}metric_resnet_rs_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    metric_image_path = f"{dir_path}metric_resnet_rs_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    metric_image_path = f"{dir_path}metric_resnet_rs_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    metric_image_path = f"{dir_path}metric_resnet_rs_aug.jpg"
        else:
            if self.mode.get() == "С тишиной":
                if self.experiment.get() == "Mel":
                    metric_image_path = f"{dir_path}metric_efnet_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    metric_image_path = f"{dir_path}metric_efnet_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    metric_image_path = f"{dir_path}metric_efnet_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    metric_image_path = f"{dir_path}metric_efnet_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    metric_image_path = f"{dir_path}metric_efnet_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    metric_image_path = f"{dir_path}metric_efnet_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    metric_image_path = f"{dir_path}metric_efnet_aug.jpg"
            else:
                if self.experiment.get() == "Mel":
                    metric_image_path = f"{dir_path}metric_efnet_rs_mel.png"
                elif self.experiment.get() == "Spectral Contrast":
                    metric_image_path = f"{dir_path}metric_efnet_rs_spectral.jpg"
                elif self.experiment.get() == "Constant Q Transform":
                    metric_image_path = f"{dir_path}metric_efnet_rs_cqt.jpg"
                elif self.experiment.get() == "Short Time Fourier Transform":
                    metric_image_path = f"{dir_path}metric_efnet_rs_stft.jpg"
                elif self.experiment.get() == "Chroma Contrast":
                    metric_image_path = f"{dir_path}metric_efnet_rs_chroma.jpg"
                elif self.experiment.get() == "Label Smoothing":
                    metric_image_path = f"{dir_path}metric_efnet_rs_label.jpg"
                elif self.experiment.get() == "Augmentations":
                    metric_image_path = f"{dir_path}metric_efnet_rs_aug.jpg"

        metric_image = Image.open(metric_image_path)
        metric_photo = ImageTk.PhotoImage(metric_image)
        metric_label = tk.Label(self.metrics_tab, image=metric_photo)
        metric_label.image = metric_photo  # Keep a reference to prevent garbage collection
        metric_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def display_predictions(self):
        self.model_file = self.load_model()

        for widget in self.predictions_tab.winfo_children():
            widget.destroy()

        y, sr = librosa.load(self.audio_file_path)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_img = Image.fromarray(spectrogram_db).convert('L')
        spectrogram_tensor = self.transform(spectrogram_img).unsqueeze(0).to(self.device)

        with no_grad():
            logits = self.model_file(spectrogram_tensor)
            probabilities = nn.functional.softmax(logits, dim=1)

        top_probabilities, top_classes = topk(probabilities, 5)
        top_probabilities = top_probabilities.cpu().numpy()[0]
        top_classes = top_classes.cpu().numpy()[0]

        tk.Label(self.predictions_tab, text="Top 5 predictions:").pack()

        for i in range(5):
            print(
                f"Class: {top_classes[i]}, Probability: {top_probabilities[i]:.4f},"
                f" Encoded bird: {self.birds[top_classes[i]]}")
            prediction_text = (f"Class: {top_classes[i]}, Probability: {top_probabilities[i]:.4f},"
                               f" Encoded bird: {self.birds[top_classes[i]]}")
            tk.Label(self.predictions_tab, text=prediction_text).pack()

    def load_model(self):
        if self.model.get() == "ResNet182":
            class ResNetGray182(nn.Module):
                def __init__(self, num_classes=182):
                    super(ResNetGray182, self).__init__()
                    resnet = resnet18(pretrained=False)
                    self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
                    self.bn1 = resnet.bn1
                    self.relu = resnet.relu
                    self.maxpool = resnet.maxpool
                    self.layer1 = resnet.layer1
                    self.layer2 = resnet.layer2
                    self.layer3 = resnet.layer3
                    self.layer4 = resnet.layer4
                    self.avgpool = resnet.avgpool
                    self.fc = nn.Linear(resnet.fc.in_features, num_classes)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    x = self.avgpool(x)
                    x = flatten(x, 1)
                    x = self.fc(x)
                    return x

            if self.mode.get() == "С тишиной":
                dir_models = "asserts\\models\\resnet\\default\\"
                if self.experiment.get() == "Mel":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Spectral Contrast":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_spectral.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Constant Q Transform":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_cqt.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Short Time Fourier Transform":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_stft.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Chroma Contrast":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_chroma.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Label Smoothing":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Augmentations":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
            else:
                dir_models = "asserts\\models\\resnet\\rs\\"
                if self.experiment.get() == "Mel":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Spectral Contrast":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_spectral.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Constant Q Transform":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_cqt.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Short Time Fourier Transform":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_stft.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Chroma Contrast":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_chroma.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Label Smoothing":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Augmentations":
                    model = ResNetGray182(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}ResNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model
        else:
            class EfficientNetGray(nn.Module):
                def __init__(self, num_classes=182):
                    super(EfficientNetGray, self).__init__()
                    self.model = create_model('tf_efficientnet_b0', pretrained=False, num_classes=num_classes)
                    in_channels = 1
                    weight = self.model.conv_stem.weight.mean(1, keepdim=True)
                    self.model.conv_stem = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                     bias=False)
                    self.model.conv_stem.weight = nn.Parameter(weight)

                def forward(self, x):
                    return self.model(x)

            if self.mode.get() == "С тишиной":
                dir_models = "asserts\\models\\efficientnet\\default\\"
                if self.experiment.get() == "Mel":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Spectral Contrast":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(
                        load(f"{dir_models}EfficientNet_spectral.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Constant Q Transform":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_cqt.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Short Time Fourier Transform":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_stft.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Chroma Contrast":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_chroma.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Label Smoothing":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Augmentations":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_mel.pth", map_location=self.device))
                    model.eval()
                    return model
            else:
                dir_models = "asserts\\models\\efficientnet\\rs\\"
                if self.experiment.get() == "Mel":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Spectral Contrast":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(
                        load(f"{dir_models}EfficientNet_rs_spectral.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Constant Q Transform":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_rs_cqt.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Short Time Fourier Transform":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_rs_stft.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Chroma Contrast":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(
                        load(f"{dir_models}EfficientNet_rs_chroma.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Label Smoothing":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model
                elif self.experiment.get() == "Augmentations":
                    model = EfficientNetGray(num_classes=182).to(self.device)
                    model.load_state_dict(load(f"{dir_models}EfficientNet_rs_mel.pth", map_location=self.device))
                    model.eval()
                    return model


if __name__ == "__main__":
    app = AudioAnalysisApp()
    app.mainloop()
