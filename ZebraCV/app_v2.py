import customtkinter as ctk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk, ImageDraw
import os
import cv2
from ultralytics import YOLO
import threading
import time
import queue
import winsound

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.title("ZEBRA")
        self.geometry("1400x1000")
        self.minsize(800, 600)
        
        
        self.model = None
        self.model_path = None
        self.current_image = None
        self.image_files = []
        self.current_image_index = 0
        self.video_source = None
        self.stream_active = False
        self.confidence = 0.5
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.camera_index = 0
        self.frame_counter = 0
        
        
        self.scale_factor = 1.0
        self.scale_delta = 0.1
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.last_boxes = None
        
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        
        control_frame = ctk.CTkFrame(self, fg_color="transparent")
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        
        self.btn_load_model = ctk.CTkButton(control_frame, text="Загрузить модель", command=self.load_model)
        self.btn_load_model.pack(side="left", padx=5)
        
        self.btn_load_image = ctk.CTkButton(control_frame, text="Загрузить изображение", command=self.load_single_image)
        self.btn_load_image.pack(side="left", padx=5)
        
        self.btn_load_folder = ctk.CTkButton(control_frame, text="Загрузить папку", command=self.load_image_folder)
        self.btn_load_folder.pack(side="left", padx=5)
        
        self.btn_load_video = ctk.CTkButton(control_frame, text="Загрузить видео", command=self.load_video)
        self.btn_load_video.pack(side="left", padx=5)
        
        
        usb_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        usb_frame.pack(side="left", padx=5)
        
        self.camera_index_var = StringVar(value="0")
        self.camera_entry = ctk.CTkEntry(usb_frame, textvariable=self.camera_index_var, width=40)
        self.camera_entry.pack(side="left", padx=5)
        
        self.btn_connect_usb = ctk.CTkButton(
            usb_frame, 
            text="USB Камера", 
            command=self.connect_usb_camera,
            fg_color="#5cb85c"
        )
        self.btn_connect_usb.pack(side="left", padx=5)
        
        
        rtsp_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        rtsp_frame.pack(side="left", padx=5)
        
        self.rtsp_url = StringVar(value="rtsp://login:password@ip:port/stream")
        self.rtsp_entry = ctk.CTkEntry(rtsp_frame, textvariable=self.rtsp_url, width=250)
        self.rtsp_entry.pack(side="left", padx=5)
        
        self.btn_connect_rtsp = ctk.CTkButton(
            rtsp_frame, 
            text="RTSP", 
            command=self.connect_rtsp,
            fg_color="#5bc0de"
        )
        self.btn_connect_rtsp.pack(side="left", padx=5)
        
        self.btn_clear = ctk.CTkButton(control_frame, text="Очистить", command=self.clear_all, fg_color="#d9534f")
        self.btn_clear.pack(side="right", padx=5)
        
        
        confidence_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        confidence_frame.pack(side="right", padx=10)
        
        ctk.CTkLabel(confidence_frame, text="Confidence:").pack(side="left")
        self.confidence_slider = ctk.CTkSlider(
            confidence_frame, 
            from_=0.1, 
            to=0.9, 
            number_of_steps=8,
            command=self.update_confidence
        )
        self.confidence_slider.pack(side="left", padx=5)
        self.confidence_slider.set(self.confidence)
        self.confidence_label = ctk.CTkLabel(confidence_frame, text=f"{self.confidence:.1f}")
        self.confidence_label.pack(side="left", padx=5)
        
        
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = ctk.CTkCanvas(self.image_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.scroll_y = ctk.CTkScrollbar(self.image_frame, orientation="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        
        self.scroll_x = ctk.CTkScrollbar(self.image_frame, orientation="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")
        
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        
        
        self.image_container = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        
        
        self.image_label = ctk.CTkLabel(self.image_container, text="Изображение не загружено", fg_color="transparent")
        self.image_label.pack(expand=True)
        
        
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        
        
        nav_frame = ctk.CTkFrame(self)
        nav_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        self.btn_prev = ctk.CTkButton(nav_frame, text="← Назад", command=self.prev_image, width=100)
        self.btn_prev.pack(side="left", padx=20)
        
        self.btn_detect = ctk.CTkButton(nav_frame, text="Старт детекции", command=self.start_detection)
        self.btn_detect.pack(side="left", padx=20, expand=True)
        
        self.btn_stop = ctk.CTkButton(
            nav_frame, 
            text="Стоп", 
            command=self.stop_stream,
            fg_color="#d9534f",
            state="disabled"
        )
        self.btn_stop.pack(side="left", padx=20)
        
        self.btn_next = ctk.CTkButton(nav_frame, text="Вперед →", command=self.next_image, width=100)
        self.btn_next.pack(side="right", padx=20)
        
        
        self.result_text = ctk.CTkTextbox(self, height=150)
        self.result_text.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        
        self.status_bar = ctk.CTkLabel(self, text="Готов к работе", anchor="w")
        self.status_bar.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        
        self.after(50, self.update_image_display)
    
    def _on_canvas_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._center_image()
    
    def _on_mousewheel(self, event):
        if self.current_image is None:
            return
        
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.scale_factor = max(self.min_scale, self.scale_factor - self.scale_delta)
        else:
            self.scale_factor = min(self.max_scale, self.scale_factor + self.scale_delta)
        
        
        if self.current_image:
            self.show_image(self.current_image, self.last_boxes)
    
    def _center_image(self):
        """Центрирует изображение на Canvas"""
        if not hasattr(self, 'image_tk'):
            return
            
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        
        img_width = self.image_tk.width()
        img_height = self.image_tk.height()
        
        
        x = max(0, (canvas_width - img_width) // 2)
        y = max(0, (canvas_height - img_height) // 2)
        
        
        self.canvas.coords(self.image_container, x, y)
    
    def update_confidence(self, value):
        self.confidence = round(float(value), 1)
        self.confidence_label.configure(text=f"{self.confidence:.1f}")
    
    def update_status(self, message):
        self.status_bar.configure(text=message)
        self.update()
    
    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("YOLO models", "*.pt")])
        if file_path:
            try:
                self.update_status("Загрузка модели...")
                self.model = YOLO(file_path)
                self.model_path = file_path
                self.result_text.insert("end", f"Модель загружена: {os.path.basename(file_path)}\n")
                self.update_status("Модель загружена")
            except Exception as e:
                self.result_text.insert("end", f"Ошибка загрузки модели: {str(e)}\n")
                self.update_status("Ошибка загрузки модели")
    
    def load_single_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.stop_stream()
            self.image_files = [file_path]
            self.current_image_index = 0
            self.load_current_image()
    
    def load_image_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.stop_stream()
            self.update_status("Загрузка изображений...")
            self.image_files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if self.image_files:
                self.current_image_index = 0
                self.load_current_image()
                self.update_status(f"Загружено {len(self.image_files)} изображений")
            else:
                self.update_status("В папке нет изображений")
    
    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.stop_stream()
            self.video_source = file_path
            self.result_text.insert("end", f"Видео загружено: {os.path.basename(file_path)}\n")
            self.update_status("Видео загружено, нажмите Старт детекции")
    
    def connect_usb_camera(self):
        self.stop_stream()
        try:
            self.camera_index = int(self.camera_index_var.get())
            self.video_source = self.camera_index
            self.result_text.insert("end", f"Подключение к USB камере (индекс {self.camera_index})...\n")
            self.update_status("Подключение к USB камере...")
            self.start_detection()
        except ValueError:
            self.result_text.insert("end", "Ошибка: индекс камеры должен быть числом\n")
            self.update_status("Ошибка индекса камеры")
    
    def connect_rtsp(self):
        rtsp_url = self.rtsp_url.get()
        if rtsp_url:
            self.stop_stream()
            self.video_source = rtsp_url
            self.result_text.insert("end", f"Подключаемся к RTSP: {rtsp_url}\n")
            self.update_status("Подключение к RTSP...")
            self.start_detection()
    
    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            try:
                self.current_image = Image.open(self.image_files[self.current_image_index])
                self.scale_factor = 1.0  
                self.show_image(self.current_image)
                self.result_text.insert("end", f"Изображение {self.current_image_index+1}/{len(self.image_files)}: {os.path.basename(self.image_files[self.current_image_index])}\n")
                self.update_status(f"Изображение {self.current_image_index+1}/{len(self.image_files)}")
            except Exception as e:
                self.result_text.insert("end", f"Ошибка загрузки изображения: {str(e)}\n")
                self.update_status("Ошибка загрузки изображения")
    
    def show_image(self, image, boxes=None):
        """Отображает изображение с возможными bounding boxes"""
        self.last_boxes = boxes  
        display_image = image.copy()
        
        
        if boxes is not None:
            draw = ImageDraw.Draw(display_image)
            for box in boxes:
                
                scaled_box = [
                    box[0] * self.scale_factor,
                    box[1] * self.scale_factor,
                    box[2] * self.scale_factor,
                    box[3] * self.scale_factor
                ]
                draw.rectangle(scaled_box, outline="red", width=3)
        
        
        new_width = int(display_image.width * self.scale_factor)
        new_height = int(display_image.height * self.scale_factor)
        display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        
        self.image_tk = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=self.image_tk, text="")
        
        
        self.image_container.configure(width=new_width, height=new_height)
        
        
        self._center_image()
        self._on_canvas_configure()
    
    def clear_all(self):
        self.stop_stream()
        self.current_image = None
        self.image_files = []
        self.current_image_index = 0
        self.video_source = None
        self.scale_factor = 1.0
        self.image_label.configure(image=None, text="Изображение не загружено")
        self.result_text.delete("1.0", "end")
        self.update_status("Готов к работе")
        if hasattr(self, 'image_tk'):
            del self.image_tk
    
    def prev_image(self):
        if self.image_files and not self.stream_active:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.load_current_image()
    
    def next_image(self):
        if self.image_files and not self.stream_active:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.load_current_image()
    
    def start_detection(self):
        if not self.model:
            self.result_text.insert("end", "Сначала загрузите модель!\n")
            return
        
        if self.video_source is not None:
            if isinstance(self.video_source, int):
                self.process_usb_camera()
            elif isinstance(self.video_source, str):
                if self.video_source.startswith('rtsp://'):
                    self.process_rtsp()
                else:
                    self.process_video()
        elif self.current_image:
            self.process_image()
        else:
            self.result_text.insert("end", "Сначала загрузите изображение или выберите источник видео!\n")
    
    def process_image(self):
        try:
            self.update_status("Идет детекция...")
            img = self.current_image.convert('RGB')
            results = self.model.predict(
                source=img,
                conf=self.confidence,
                imgsz=640
            )
            
            boxes = []
            drone_detected = False
            self.result_text.insert("end", "\nРезультаты детекции:\n")
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    boxes.append(xyxy)
                    class_id = int(box.cls)
                    conf = float(box.conf)
                    if class_id == 0:
                        drone_detected = True
                    self.result_text.insert("end", 
                        f"Объект: {result.names[class_id]} | "
                        f"Уверенность: {conf:.2f} | "
                        f"Координаты: {[round(x, 1) for x in xyxy]}\n")
            if drone_detected:
                winsound.Beep(1000, 500)
            self.show_image(img, boxes)
            self.update_status(f"Детекция завершена ({len(boxes)} объектов)")
            
        except Exception as e:
            self.result_text.insert("end", f"Ошибка детекции: {str(e)}\n")
            self.update_status("Ошибка детекции")
    
    def process_usb_camera(self):
        if not isinstance(self.video_source, int):
            return
        
        self.stream_active = True
        self.btn_stop.configure(state="normal")
        self.btn_detect.configure(state="disabled")
        self.frame_counter = 0
        
        threading.Thread(target=self._usb_camera_thread, daemon=True).start()
    
    def process_video(self):
        if not isinstance(self.video_source, str):
            return
        
        self.stream_active = True
        self.btn_stop.configure(state="normal")
        self.btn_detect.configure(state="disabled")
        self.frame_counter = 0
        
        threading.Thread(target=self._video_processing_thread, daemon=True).start()
    
    def process_rtsp(self):
        if not isinstance(self.video_source, str):
            return
        
        self.stream_active = True
        self.btn_stop.configure(state="normal")
        self.btn_detect.configure(state="disabled")
        self.frame_counter = 0
        
        threading.Thread(target=self._rtsp_processing_thread, daemon=True).start()
    
    def _usb_camera_thread(self):
        cap = cv2.VideoCapture(self.video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.stream_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.after(0, lambda: self.result_text.insert("end", "Ошибка чтения с USB камеры\n"))
                time.sleep(1)
                continue
            
            self._process_frame(frame)
            time.sleep(0.03)  
        
        cap.release()
        self.after(0, self._stream_processing_finished)
    
    def _video_processing_thread(self):
        cap = cv2.VideoCapture(self.video_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / fps if fps > 0 else 0.03
        
        while self.stream_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self._process_frame(frame)
            time.sleep(frame_delay)
        
        cap.release()
        self.after(0, self._stream_processing_finished)
    
    def _rtsp_processing_thread(self):
        cap = cv2.VideoCapture(self.video_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.stream_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.after(0, lambda: self.result_text.insert("end", "Ошибка чтения RTSP потока\n"))
                time.sleep(1)
                continue
            
            self._process_frame(frame)
            time.sleep(0.03)  
        
        cap.release()
        self.after(0, self._stream_processing_finished)
    
    def _process_frame(self, frame):
        try:
            self.frame_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            results = self.model.predict(
                source=img,
                conf=self.confidence,
                imgsz=640,
                verbose=False
            )
            
            boxes = []
            detection_info = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    boxes.append(xyxy)
                    class_id = int(box.cls)
                    conf = float(box.conf)
                    if class_id == 0:
                        drone_detected = True
                    detection_info.append(
                        f"Кадр {self.frame_counter}: Объект: {result.names[class_id]} | "
                        f"Уверенность: {conf:.2f} | "
                        f"Координаты: {[round(x, 1) for x in xyxy]}\n"
                    )
            if drone_detected:
                winsound.Beep(1000, 300)        
            
            
            if self.frame_queue.empty():
                self.frame_queue.put((img, boxes, detection_info))
        
        except Exception as e:
            self.after(0, lambda: self.result_text.insert("end", f"Ошибка обработки кадра: {str(e)}\n"))
    
    def update_image_display(self):
        try:
            if not self.frame_queue.empty():
                img, boxes, detection_info = self.frame_queue.get_nowait()
                self.current_image = img
                self.scale_factor = 1.0  
                self.show_image(img, boxes)
                
                
                for info in detection_info:
                    self.result_text.insert("end", info)
                
                if self.stream_active:
                    source_type = "USB камера" if isinstance(self.video_source, int) else \
                                "RTSP" if isinstance(self.video_source, str) and self.video_source.startswith('rtsp://') else \
                                "Видеофайл"
                    self.update_status(f"Обработка {source_type}... Нажмите 'Стоп' для остановки")
        
        except queue.Empty:
            pass
        
        self.after(50, self.update_image_display)
    
    def _stream_processing_finished(self):
        self.stream_active = False
        self.btn_stop.configure(state="disabled")
        self.btn_detect.configure(state="normal")
        self.update_status("Обработка завершена")
    
    def stop_stream(self):
        self.stream_active = False
        self.btn_stop.configure(state="disabled")
        self.btn_detect.configure(state="normal")
        
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
    
    def on_closing(self):
        self.stop_stream()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()