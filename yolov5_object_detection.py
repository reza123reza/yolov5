"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    YOLOv5 Object Detection System                          ║
║                    سیستم تشخیص اشیاء با YOLOv5                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  نویسنده: رضا صفری فروشانی (Reza Safari Froushani)                      ║
║  ایمیل: safarireza@gmail.com                                              ║
║  گیتهاب: https://github.com/reza123reza                                   ║
║  تاریخ: نوامبر 2024                                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  توضیحات:                                                                  ║
║  این اسکریپت برای تشخیص اشیاء با استفاده از مدل از پیش آموزش‌دیده      ║
║  YOLOv5 طراحی شده است.                                                    ║
║                                                                            ║
║  قابلیت‌ها:                                                                ║
║  ✓ نصب خودکار تمام پکیج‌های مورد نیاز                                    ║
║  ✓ دانلود و راه‌اندازی YOLOv5                                            ║
║  ✓ تشخیص اشیاء در تصاویر ثابت                                           ║
║  ✓ تشخیص اشیاء در ویدیوهای ضبط شده                                     ║
║  ✓ تشخیص اشیاء از وبکم (زنده)                                           ║
║  ✓ فیلتر کردن کلاس‌های خاص                                              ║
║  ✓ تنظیم آستانه اطمینان                                                  ║
║  ✓ ذخیره خودکار نتایج                                                    ║
║  ✓ نمایش نتایج در محیط نوتبوک                                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║  نحوه استفاده:                                                            ║
║  python yolov5_object_detection.py                                         ║
║                                                                            ║
║  یا به صورت ماژول:                                                        ║
║  from yolov5_object_detection import YOLOv5Detector                        ║
║  detector = YOLOv5Detector()                                               ║
║  detector.setup_environment()                                              ║
║  detector.detect_image("path/to/image.jpg")                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
║  کپی برداری از این کد بدون ذکر منبع و نام نویسنده ممنوع است.            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Union


class YOLOv5Detector:
    """
    کلاس اصلی برای تشخیص اشیاء با YOLOv5
    
    این کلاس تمام عملیات مورد نیاز برای تشخیص اشیاء با استفاده از
    مدل‌های از پیش آموزش‌دیده YOLOv5 را فراهم می‌کند.
    
    نویسنده: رضا صفری فروشانی
    ایمیل: safarireza@gmail.com
    گیتهاب: https://github.com/reza123reza
    
    Attributes:
        weights (str): مسیر یا نام فایل وزن‌های مدل
        conf_threshold (float): آستانه اطمینان برای تشخیص (0-1)
        yolov5_path (Path): مسیر پوشه YOLOv5
    """
    
    def __init__(self, weights: str = 'yolov5m.pt', conf_threshold: float = 0.7):
        """
        مقداردهی اولیه کلاس تشخیص‌دهنده YOLOv5
        
        Args:
            weights: نام یا مسیر فایل وزن‌های مدل
                    گزینه‌ها: yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
            conf_threshold: آستانه اطمینان (بین 0 تا 1)
        
        نوشته شده توسط: رضا صفری فروشانی
        """
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.yolov5_path = None
        self.original_dir = os.getcwd()
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "YOLOv5 Object Detector" + " " * 35 + "║")
        print("║" + " " * 78 + "║")
        print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
        print("║" + "  ایمیل: safarireza@gmail.com".ljust(77) + " ║")
        print("║" + "  گیتهاب: https://github.com/reza123reza".ljust(77) + " ║")
        print("╚" + "═" * 78 + "╝")
        
    def setup_environment(self) -> bool:
        """
        نصب و راه‌اندازی محیط YOLOv5
        
        این متد تمام پکیج‌های لازم را نصب کرده و YOLOv5 را دانلود می‌کند.
        
        Returns:
            bool: True در صورت موفقیت، False در صورت خطا
            
        توسعه‌دهنده: رضا صفری فروشانی (safarireza@gmail.com)
        """
        try:
            print("\n" + "═" * 80)
            print("شروع راه‌اندازی محیط YOLOv5")
            print("توسعه‌دهنده: رضا صفری فروشانی")
            print("═" * 80)
            
            # مرحله 1: آپگرید pip
            print("\n[1/4] در حال آپگرید pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("      ✓ pip با موفقیت آپگرید شد")
            else:
                print("      ⚠ خطا در آپگرید pip (ادامه می‌دهیم...)")
            
            # مرحله 2: نصب TensorBoard
            # (TensorFlow لازم نیست - YOLOv5 فقط با PyTorch کار می‌کنه)
            print("\n[2/4] در حال نصب TensorBoard...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tensorboard", "--quiet"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("      ✓ TensorBoard نصب شد")
            
            # مرحله 3: نصب PyTorch
            print("\n[3/4] در حال نصب PyTorch...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "--quiet"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("      ✓ PyTorch نصب شد")
            else:
                print("      ⚠ خطا در نصب PyTorch")
            
            # مرحله 4: دانلود YOLOv5
            print("\n[4/4] در حال دانلود YOLOv5...")
            if os.path.exists('yolov5'):
                print("      ℹ پوشه yolov5 از قبل وجود دارد")
                user_input = input("      آیا می‌خواهید دوباره دانلود شود؟ (y/n): ")
                if user_input.lower() == 'y':
                    shutil.rmtree('yolov5')
                    subprocess.run(
                        ['git', 'clone', 'https://github.com/ultralytics/yolov5'],
                        capture_output=True
                    )
                    print("      ✓ YOLOv5 دانلود شد")
            else:
                result = subprocess.run(
                    ['git', 'clone', 'https://github.com/ultralytics/yolov5'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("      ✓ YOLOv5 دانلود شد")
                else:
                    print("      ✗ خطا در دانلود YOLOv5")
                    return False
            
            # تنظیم مسیر YOLOv5
            self.yolov5_path = Path(os.path.abspath('yolov5'))
            os.chdir(self.yolov5_path)
            
            # نصب requirements
            print("\n      در حال نصب requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                capture_output=True
            )
            print("      ✓ Requirements نصب شد")
            
            # بازگشت به مسیر اصلی
            os.chdir(self.original_dir)
            
            print("\n" + "═" * 80)
            print("✓ راه‌اندازی با موفقیت کامل شد!")
            print(f"✓ مسیر YOLOv5: {self.yolov5_path}")
            print("═" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n✗ خطا در راه‌اندازی: {str(e)}")
            os.chdir(self.original_dir)
            return False
    
    def _build_cmd(self, source: str, classes: Optional[List[int]], save_dir: str) -> List[str]:
        """دستور detect.py؛ نتایج توی save_dir (نسبت به پوشه‌ی کاری فعلی) ذخیره می‌شن"""
        cmd = [
            sys.executable, "detect.py",
            "--weights", self.weights,
            "--conf-thres", str(self.conf_threshold),
            "--source", source,
            "--project", os.path.join(self.original_dir, save_dir),
        ]
        if classes is not None:
            cmd.extend(["--classes"] + [str(c) for c in classes])
            print(f"🎯 کلاس‌های فیلتر شده: {classes}")
        return cmd

    def _latest_run_dir(self, save_dir: str) -> Optional[Path]:
        """جدیدترین پوشه‌ی exp بر اساس زمان ساخت (نه ترتیب الفبایی - exp10 قبل از exp2 میاد)"""
        detect_path = Path(self.original_dir) / save_dir
        if not detect_path.is_dir():
            return None
        runs = [d for d in detect_path.iterdir() if d.is_dir()]
        return max(runs, key=lambda d: d.stat().st_mtime) if runs else None

    def detect_image(self, 
                    image_path: str, 
                    classes: Optional[List[int]] = None,
                    save_dir: str = 'runs/detect',
                    show_result: bool = True) -> Optional[str]:
        """
        تشخیص اشیاء در یک تصویر
        
        این متد از مدل YOLOv5 برای تشخیص اشیاء در تصویر ورودی استفاده می‌کند.
        
        Args:
            image_path: مسیر تصویر ورودی
            classes: لیست شماره کلاس‌های مورد نظر (None = همه کلاس‌ها)
                    مثال: [0, 2] برای person و car
            save_dir: مسیر ذخیره نتایج
            show_result: نمایش نتیجه (در محیط نوتبوک)
        
        Returns:
            str: مسیر تصویر خروجی یا None در صورت خطا
        
        کد نوشته شده توسط: رضا صفری فروشانی
        GitHub: https://github.com/reza123reza
        """
        if self.yolov5_path is None:
            print("✗ لطفاً ابتدا setup_environment() را اجرا کنید")
            return None
        
        if not os.path.exists(image_path):
            print(f"✗ تصویر پیدا نشد: {image_path}")
            return None
        
        print(f"\n{'═'*80}")
        print(f"تشخیص اشیاء در تصویر")
        print(f"نویسنده: رضا صفری فروشانی (safarireza@gmail.com)")
        print(f"{'═'*80}")
        print(f"📷 تصویر ورودی: {image_path}")
        print(f"⚙️  وزن مدل: {self.weights}")
        print(f"📊 آستانه اطمینان: {self.conf_threshold}")
        
        try:
            os.chdir(self.yolov5_path)
            
            # ساخت دستور
            cmd = self._build_cmd(os.path.abspath(image_path), classes, save_dir)
            
            print(f"\n⏳ در حال پردازش...")
            
            # اجرای تشخیص
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ تشخیص با موفقیت انجام شد!")
                
                # پیدا کردن آخرین فولدر exp
                latest_exp = self._latest_run_dir(save_dir)
                if latest_exp is not None:
                    output_file = latest_exp / Path(image_path).name
                    
                    if output_file.exists():
                        print(f"💾 نتیجه ذخیره شد: {output_file}")
                        
                        # نمایش نتیجه
                        if show_result:
                            try:
                                from IPython.display import Image, display
                                print("\n📸 نمایش نتیجه:")
                                display(Image(filename=str(output_file)))
                            except ImportError:
                                print("ℹ️  برای نمایش تصویر در محیط نوتبوک قرار دهید")
                        
                        os.chdir(self.original_dir)
                        return str(output_file)
            else:
                print(f"✗ خطا: {result.stderr}")
                
        except Exception as e:
            print(f"✗ خطا در تشخیص: {str(e)}")
        finally:
            os.chdir(self.original_dir)
        
        return None
    
    def detect_video(self, 
                    video_path: str, 
                    classes: Optional[List[int]] = None,
                    save_dir: str = 'runs/detect') -> Optional[str]:
        """
        تشخیص اشیاء در یک ویدیو
        
        Args:
            video_path: مسیر ویدیو ورودی
            classes: لیست شماره کلاس‌های مورد نظر
            save_dir: مسیر ذخیره نتایج
        
        Returns:
            str: مسیر ویدیوی خروجی یا None در صورت خطا
            
        نوشته شده توسط: رضا صفری فروشانی
        Email: safarireza@gmail.com
        """
        if self.yolov5_path is None:
            print("✗ لطفاً ابتدا setup_environment() را اجرا کنید")
            return None
        
        if not os.path.exists(video_path):
            print(f"✗ ویدیو پیدا نشد: {video_path}")
            return None
        
        print(f"\n{'═'*80}")
        print(f"تشخیص اشیاء در ویدیو")
        print(f"نویسنده: رضا صفری فروشانی")
        print(f"{'═'*80}")
        print(f"🎬 ویدیو ورودی: {video_path}")
        print(f"⚙️  وزن مدل: {self.weights}")
        print(f"📊 آستانه اطمینان: {self.conf_threshold}")
        
        try:
            os.chdir(self.yolov5_path)
            
            # ساخت دستور
            cmd = self._build_cmd(os.path.abspath(video_path), classes, save_dir)
            
            print(f"\n⏳ در حال پردازش ویدیو (ممکن است زمان‌بر باشد)...")
            
            # اجرای تشخیص
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ پردازش ویدیو با موفقیت انجام شد!")
                
                # پیدا کردن ویدیوی خروجی
                latest_exp = self._latest_run_dir(save_dir)
                if latest_exp is not None:
                    video_name = Path(video_path).stem
                    output_file = list(latest_exp.glob(f"{video_name}.*"))
                    
                    if output_file:
                        print(f"💾 ویدیو ذخیره شد: {output_file[0]}")
                        os.chdir(self.original_dir)
                        return str(output_file[0])
            else:
                print(f"✗ خطا: {result.stderr}")
                
        except Exception as e:
            print(f"✗ خطا در پردازش ویدیو: {str(e)}")
        finally:
            os.chdir(self.original_dir)
        
        return None
    
    def detect_webcam(self, 
                     classes: Optional[List[int]] = None,
                     camera_index: int = 0,
                     save_dir: str = 'runs/detect'):
        """
        تشخیص اشیاء زنده از وبکم
        
        این متد تشخیص اشیاء را به صورت لحظه‌ای از دوربین وب انجام می‌دهد.
        برای خروج کلید 'q' را فشار دهید.
        
        Args:
            classes: لیست شماره کلاس‌های مورد نظر
            camera_index: شماره دوربین (معمولاً 0)
        
        توسعه یافته توسط: رضا صفری فروشانی
        https://github.com/reza123reza
        """
        if self.yolov5_path is None:
            print("✗ لطفاً ابتدا setup_environment() را اجرا کنید")
            return
        
        print(f"\n{'═'*80}")
        print(f"تشخیص اشیاء زنده از وبکم")
        print(f"نویسنده: رضا صفری فروشانی")
        print(f"{'═'*80}")
        print(f"📹 دوربین: {camera_index}")
        print(f"⚙️  وزن مدل: {self.weights}")
        print(f"📊 آستانه اطمینان: {self.conf_threshold}")
        print(f"\nℹ️  برای خروج کلید 'q' را فشار دهید")
        
        try:
            os.chdir(self.yolov5_path)
            
            # ساخت دستور
            cmd = self._build_cmd(str(camera_index), classes, save_dir)
            
            print(f"\n⏳ در حال راه‌اندازی دوربین...")
            
            # اجرای تشخیص
            subprocess.run(cmd)
            
            print("\n✓ تشخیص متوقف شد")
                
        except KeyboardInterrupt:
            print("\n\nℹ️  تشخیص توسط کاربر متوقف شد")
        except Exception as e:
            print(f"✗ خطا: {str(e)}")
        finally:
            os.chdir(self.original_dir)
    
    def get_coco_classes(self) -> dict:
        """
        دریافت لیست کلاس‌های COCO
        
        Returns:
            dict: دیکشنری شماره و نام کلاس‌ها
        
        Copyright © 2024 Reza Safari Froushani
        """
        classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        return classes
    
    def print_classes(self):
        """
        چاپ لیست تمام کلاس‌های قابل تشخیص
        
        این متد لیست کامل 80 کلاس COCO را نمایش می‌دهد.
        
        نویسنده: رضا صفری فروشانی
        """
        classes = self.get_coco_classes()
        
        print(f"\n{'═'*80}")
        print("لیست کلاس‌های قابل تشخیص (COCO Dataset)")
        print(f"{'═'*80}")
        
        for i in range(0, len(classes), 4):
            line = ""
            for j in range(4):
                if i + j < len(classes):
                    line += f"{i+j:2d}: {classes[i+j]:15s} "
            print(line)
        
        print(f"{'═'*80}")
        print(f"مجموع: {len(classes)} کلاس")
        print(f"{'═'*80}\n")


def main():
    """
    تابع اصلی برای نمایش مثال‌های استفاده
    
    این تابع نمونه‌هایی از نحوه استفاده از کلاس YOLOv5Detector را نشان می‌دهد.
    
    استفاده:
        python yolov5_object_detection.py
    
    ساخته شده توسط: رضا صفری فروشانی
    ایمیل: safarireza@gmail.com
    گیتهاب: https://github.com/reza123reza
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "YOLOv5 Object Detection - نمونه استفاده" + " " * 22 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
    print("║" + "  ایمیل: safarireza@gmail.com".ljust(77) + " ║")
    print("║" + "  گیتهاب: https://github.com/reza123reza".ljust(77) + " ║")
    print("╚" + "═" * 78 + "╝")
    
    # ساخت شیء تشخیص‌دهنده
    detector = YOLOv5Detector(weights='yolov5m.pt', conf_threshold=0.7)
    
    # راه‌اندازی محیط
    if not detector.setup_environment():
        print("✗ خطا در راه‌اندازی محیط")
        return
    
    # نمایش کلاس‌ها
    detector.print_classes()
    
    # منوی تعاملی
    while True:
        print("\n" + "═" * 80)
        print("منوی اصلی:")
        print("1. تشخیص اشیاء در تصویر")
        print("2. تشخیص اشیاء در ویدیو")
        print("3. تشخیص زنده از وبکم")
        print("4. نمایش لیست کلاس‌ها")
        print("5. خروج")
        print("═" * 80)
        
        choice = input("\nانتخاب کنید (1-5): ").strip()
        
        if choice == '1':
            image_path = input("مسیر تصویر را وارد کنید: ").strip()
            filter_choice = input("آیا می‌خواهید کلاس خاصی را فیلتر کنید؟ (y/n): ").strip().lower()
            
            classes = None
            if filter_choice == 'y':
                class_input = input("شماره کلاس‌ها را با کاما جدا کنید (مثال: 0,2,3): ").strip()
                try:
                    classes = [int(x.strip()) for x in class_input.split(',')]
                except:
                    print("⚠ فرمت نادرست. همه کلاس‌ها تشخیص داده می‌شوند.")
            
            detector.detect_image(image_path, classes=classes)
        
        elif choice == '2':
            video_path = input("مسیر ویدیو را وارد کنید: ").strip()
            filter_choice = input("آیا می‌خواهید کلاس خاصی را فیلتر کنید؟ (y/n): ").strip().lower()
            
            classes = None
            if filter_choice == 'y':
                class_input = input("شماره کلاس‌ها را با کاما جدا کنید: ").strip()
                try:
                    classes = [int(x.strip()) for x in class_input.split(',')]
                except:
                    print("⚠ فرمت نادرست. همه کلاس‌ها تشخیص داده می‌شوند.")
            
            detector.detect_video(video_path, classes=classes)
        
        elif choice == '3':
            filter_choice = input("آیا می‌خواهید کلاس خاصی را فیلتر کنید؟ (y/n): ").strip().lower()
            
            classes = None
            if filter_choice == 'y':
                class_input = input("شماره کلاس‌ها را با کاما جدا کنید: ").strip()
                try:
                    classes = [int(x.strip()) for x in class_input.split(',')]
                except:
                    print("⚠ فرمت نادرست. همه کلاس‌ها تشخیص داده می‌شوند.")
            
            detector.detect_webcam(classes=classes)
        
        elif choice == '4':
            detector.print_classes()
        
        elif choice == '5':
            print("\n" + "═" * 80)
            print("خروج از برنامه...")
            print("با تشکر از استفاده")
            print("نویسنده: رضا صفری فروشانی")
            print("═" * 80)
            break
        
        else:
            print("⚠ انتخاب نامعتبر!")


if __name__ == "__main__":
    """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
    ║  ایمیل: safarireza@gmail.com                                              ║
    ║  گیتهاب: https://github.com/reza123reza                                   ║
    ║                                                                            ║
    ║  این کد تحت حق نشر محفوظ است.                                            ║
    ║  کپی برداری، توزیع یا تغییر بدون اجازه کتبی نویسنده ممنوع است.           ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    main()
