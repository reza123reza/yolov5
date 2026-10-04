# 🎯 YOLOv5 Object Detection System
# سیستم تشخیص اشیاء با YOLOv5

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Latest-green.svg)
![License](https://img.shields.io/badge/License-Copyright-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**یک سیستم کامل و حرفه‌ای برای تشخیص اشیاء با استفاده از مدل‌های از پیش آموزش‌دیده YOLOv5**

[English](#english) | [فارسی](#فارسی)

</div>

---

## 📋 فهرست مطالب

- [درباره پروژه](#-درباره-پروژه)
- [ویژگی‌ها](#-ویژگی‌ها)
- [نصب و راه‌اندازی](#-نصب-و-راه‌اندازی)
- [نحوه استفاده](#-نحوه-استفاده)
- [مثال‌ها](#-مثال‌ها)
- [مدل‌های موجود](#-مدل‌های-موجود)
- [کلاس‌های قابل تشخیص](#-کلاس‌های-قابل-تشخیص)
- [نویسنده](#-نویسنده)
- [مجوز](#-مجوز)

---

## 🎓 درباره پروژه

این پروژه یک سیستم کامل و کاربرپسند برای **تشخیص اشیاء** با استفاده از مدل‌های از پیش آموزش‌دیده **YOLOv5** است. این کد به صورت تخصصی برای استفاده آسان و سریع طراحی شده و تمام پیچیدگی‌های نصب و راه‌اندازی را خودکار انجام می‌دهد.

### 🌟 چرا این پروژه؟

- ✅ **نصب خودکار**: تمام پکیج‌های لازم به صورت خودکار نصب می‌شوند
- ✅ **رابط کاربری ساده**: منوی تعاملی و کاربرپسند
- ✅ **انعطاف‌پذیر**: قابل استفاده به صورت اسکریپت یا ماژول
- ✅ **مستندسازی کامل**: کامنت‌های فارسی و توضیحات جامع
- ✅ **چند منظوره**: تشخیص در تصویر، ویدیو و وبکم

---

## ⚡ ویژگی‌ها

### 🔧 قابلیت‌های فنی

- **نصب خودکار پکیج‌ها**: PyTorch, TensorBoard, YOLOv5
- **دانلود خودکار YOLOv5**: از مخزن رسمی Ultralytics
- **تشخیص در تصویر**: پردازش تصاویر ثابت با دقت بالا
- **تشخیص در ویدیو**: پردازش ویدیوهای ضبط شده
- **تشخیص زنده**: استفاده از وبکم برای تشخیص لحظه‌ای
- **فیلتر کلاس**: تشخیص کلاس‌های خاص (مثلاً فقط انسان و ماشین)
- **تنظیم آستانه اطمینان**: کنترل دقت تشخیص
- **ذخیره خودکار**: ذخیره نتایج در پوشه‌های مجزا
- **نمایش نتایج**: نمایش خودکار در محیط نوتبوک

### 🎨 ویژگی‌های کاربری

- **منوی تعاملی**: رابط کاربری ساده و کاربرپسند
- **پیام‌های فارسی**: تمام پیام‌ها به زبان فارسی
- **راهنمای استفاده**: مثال‌های کاربردی و کامل
- **مدیریت خطا**: مدیریت حرفه‌ای خطاها و پیام‌های راهنما

---

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها

```bash
# Python 3.7 یا بالاتر
python --version

# Git
git --version
```

### نصب

#### روش 1: استفاده از Git

```bash
# کلون کردن ریپازیتوری
git clone https://github.com/reza123reza/yolov5-object-detection.git
cd yolov5-object-detection

# اجرای برنامه (نصب خودکار انجام می‌شود)
python yolov5_object_detection.py
```

#### روش 2: دانلود مستقیم

1. فایل `yolov5_object_detection.py` را دانلود کنید
2. در ترمینال اجرا کنید:

```bash
python yolov5_object_detection.py
```

### راه‌اندازی اولیه

برنامه به صورت خودکار موارد زیر را نصب می‌کند:

- ✅ آپگرید pip
- ✅ نصب TensorBoard
- ✅ نصب PyTorch
- ✅ دانلود YOLOv5 از GitHub
- ✅ نصب requirements

---

## 📖 نحوه استفاده

### استفاده به صورت اسکریپت

```bash
python yolov5_object_detection.py
```

سپس از منوی تعاملی استفاده کنید:

```
═══════════════════════════════════════════════════════════
منوی اصلی:
1. تشخیص اشیاء در تصویر
2. تشخیص اشیاء در ویدیو
3. تشخیص زنده از وبکم
4. نمایش لیست کلاس‌ها
5. خروج
═══════════════════════════════════════════════════════════

انتخاب کنید (1-5): 
```

### استفاده به صورت ماژول

```python
from yolov5_object_detection import YOLOv5Detector

# ساخت شیء تشخیص‌دهنده
detector = YOLOv5Detector(weights='yolov5m.pt', conf_threshold=0.7)

# راه‌اندازی محیط (فقط بار اول)
detector.setup_environment()

# تشخیص در تصویر
detector.detect_image("path/to/image.jpg")

# تشخیص در ویدیو
detector.detect_video("path/to/video.mp4")

# تشخیص زنده از وبکم
detector.detect_webcam()

# تشخیص با فیلتر کلاس (فقط انسان و ماشین)
detector.detect_image("image.jpg", classes=[0, 2])
```

---

## 💡 مثال‌ها

### مثال 1: تشخیص همه اشیاء در تصویر

```python
from yolov5_object_detection import YOLOv5Detector

detector = YOLOv5Detector()
detector.setup_environment()
detector.detect_image("my_photo.jpg")
```

### مثال 2: تشخیص فقط انسان‌ها

```python
detector = YOLOv5Detector(conf_threshold=0.5)
detector.setup_environment()

# کلاس 0 = person
detector.detect_image("street.jpg", classes=[0])
```

### مثال 3: تشخیص در ویدیو با آستانه بالا

```python
detector = YOLOv5Detector(
    weights='yolov5l.pt',  # مدل بزرگ‌تر برای دقت بیشتر
    conf_threshold=0.8
)
detector.setup_environment()
detector.detect_video("traffic.mp4")
```

### مثال 4: تشخیص وسایل نقلیه

```python
detector = YOLOv5Detector()
detector.setup_environment()

# 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck
vehicles = [2, 3, 5, 6, 7]
detector.detect_image("parking.jpg", classes=vehicles)
```

### مثال 5: تشخیص زنده از وبکم

```python
detector = YOLOv5Detector(conf_threshold=0.6)
detector.setup_environment()

# تشخیص فقط افراد و موبایل
detector.detect_webcam(classes=[0, 67])
```

---

## 🎯 مدل‌های موجود

| مدل | حجم | سرعت | دقت mAP | موارد استفاده |
|-----|------|------|---------|---------------|
| **yolov5s.pt** | 14 MB | ⚡⚡⚡⚡⚡ | 37.4 | دستگاه‌های ضعیف، سرعت بالا |
| **yolov5m.pt** | 40 MB | ⚡⚡⚡⚡ | 45.4 | استفاده عمومی (پیشنهادی) |
| **yolov5l.pt** | 89 MB | ⚡⚡⚡ | 49.0 | دقت بالا |
| **yolov5x.pt** | 166 MB | ⚡⚡ | 50.7 | حداکثر دقت |

### انتخاب مدل مناسب

```python
# سرعت بالا (مناسب وبکم و پردازش زنده)
detector = YOLOv5Detector(weights='yolov5s.pt')

# تعادل بین سرعت و دقت (پیشنهادی)
detector = YOLOv5Detector(weights='yolov5m.pt')

# دقت بالا (مناسب تحلیل دقیق)
detector = YOLOv5Detector(weights='yolov5l.pt')

# حداکثر دقت (مناسب تحقیقات)
detector = YOLOv5Detector(weights='yolov5x.pt')
```

---

## 📊 کلاس‌های قابل تشخیص

این سیستم می‌تواند **80 کلاس مختلف** از دیتاست COCO را تشخیص دهد:

<details>
<summary>نمایش لیست کامل کلاس‌ها (کلیک کنید)</summary>

| شماره | نام | شماره | نام | شماره | نام | شماره | نام |
|-------|-----|-------|-----|-------|-----|-------|-----|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

</details>

### استفاده از کلاس‌ها

```python
# دریافت لیست کلاس‌ها
detector = YOLOv5Detector()
classes = detector.get_coco_classes()

# چاپ لیست کلاس‌ها
detector.print_classes()

# تشخیص کلاس‌های خاص
detector.detect_image("image.jpg", classes=[0, 16, 17])  # انسان، سگ، اسب
```

---

## 🎨 ساختار پروژه

```
yolov5-object-detection/
│
├── yolov5_object_detection.py    # فایل اصلی برنامه
├── example_usage.py               # مثال‌های آماده
├── INSTRUCTIONS_FA.txt            # راهنمای کامل فارسی
├── README.md                      # این فایل
├── LICENSE                        # مجوز استفاده
├── requirements.txt               # پکیج‌های مورد نیاز (اختیاری)
│
├── runs/                          # نتایج تشخیص (پیش‌فرض save_dir)
│   └── detect/
│       ├── exp/
│       ├── exp2/
│       └── ...
│
└── yolov5/                        # پوشه YOLOv5 (خودکار دانلود می‌شود)
    ├── detect.py
    ├── train.py
    ├── models/
    └── utils/
```

---

## ⚙️ تنظیمات پیشرفته

### تغییر آستانه اطمینان

```python
# آستانه پایین = تشخیص بیشتر ولی احتمال خطای بیشتر
detector = YOLOv5Detector(conf_threshold=0.4)

# آستانه بالا = تشخیص کمتر ولی دقت بیشتر
detector = YOLOv5Detector(conf_threshold=0.9)
```

### استفاده از دوربین خاص

```python
# دوربین اول (پیش‌فرض)
detector.detect_webcam(camera_index=0)

# دوربین دوم
detector.detect_webcam(camera_index=1)
```

### ذخیره در مسیر سفارشی

```python
detector.detect_image("image.jpg", save_dir="my_results/")
```

---

## 🐛 رفع مشکلات

### مشکل 1: خطای نصب PyTorch

**راه حل:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### مشکل 2: خطای Git

**راه حل:**
- مطمئن شوید Git نصب است: `git --version`
- یا YOLOv5 را دستی دانلود کنید از [اینجا](https://github.com/ultralytics/yolov5)

### مشکل 3: خطای وبکم

**راه حل:**
- دسترسی دوربین را بررسی کنید
- از camera_index متفاوت استفاده کنید: `detect_webcam(camera_index=1)`

### مشکل 4: کندی پردازش

**راه حل:**
- از مدل کوچک‌تر استفاده کنید: `yolov5s.pt`
- آستانه اطمینان را بالا ببرید: `conf_threshold=0.8`

---

## 📈 بهبودهای آینده

- [ ] پشتیبانی از مدل‌های سفارشی
- [ ] رابط گرافیکی (GUI)
- [ ] پشتیبانی از YOLOv8
- [ ] تشخیص چند‌نخی برای سرعت بیشتر
- [ ] خروجی به فرمت JSON
- [ ] API RESTful

---

## 👨‍💻 نویسنده

<div align="center">

### رضا صفری فروشانی
### Reza Safari Froushani

[![Email](https://img.shields.io/badge/Email-safarireza@gmail.com-red?style=for-the-badge&logo=gmail)](mailto:safarireza@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-reza123reza-black?style=for-the-badge&logo=github)](https://github.com/reza123reza)

**توسعه‌دهنده هوش مصنوعی و بینایی ماشین**

</div>

### تماس با من

- 📧 **ایمیل**: [safarireza@gmail.com](mailto:safarireza@gmail.com)
- 💻 **گیتهاب**: [github.com/reza123reza](https://github.com/reza123reza)
- 🌐 **وب‌سایت**: *(در دست طراحی)*

---

## 📄 مجوز

```
╔════════════════════════════════════════════════════════════════════════════╗
║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
║  حق نشر © ۲۰۲۴ رضا صفری فروشانی. تمامی حقوق محفوظ است.                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  این کد تحت حق نشر محفوظ است و استفاده، کپی‌برداری، توزیع یا تغییر       ║
║  آن بدون اجازه کتبی نویسنده ممنوع است.                                   ║
║                                                                            ║
║  This code is copyrighted. Use, copying, distribution or modification     ║
║  without written permission from the author is prohibited.                ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  استثنائات:                                                               ║
║  - استفاده برای اهداف آموزشی شخصی                                        ║
║  - استفاده در پروژه‌های غیرتجاری با ذکر منبع                             ║
║                                                                            ║
║  Exceptions:                                                               ║
║  - Use for personal educational purposes                                   ║
║  - Use in non-commercial projects with attribution                         ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### شرایط استفاده

1. **ذکر منبع الزامی**: در صورت استفاده از این کد، باید نام نویسنده و لینک گیتهاب ذکر شود
2. **استفاده تجاری**: نیاز به اجازه کتبی دارد
3. **تغییرات**: تغییرات باید مستندسازی شوند
4. **مسئولیت**: نویسنده مسئولیتی در قبال استفاده از این کد ندارد

---

## 🙏 تشکر و قدردانی

این پروژه با استفاده از منابع زیر ساخته شده است:

- [**YOLOv5**](https://github.com/ultralytics/yolov5) - Ultralytics
- [**PyTorch**](https://pytorch.org/) - Facebook AI Research
- [**COCO Dataset**](https://cocodataset.org/) - Microsoft

---

## ⭐ حمایت از پروژه

اگر این پروژه برایتان مفید بود:

- ⭐ یک ستاره به ریپازیتوری بدهید
- 🐛 باگ‌ها را گزارش دهید
- 💡 پیشنهادات خود را ارسال کنید
- 🔀 Pull Request بفرستید
- 📧 نظرات خود را با من به اشتراک بگذارید

---

## 📞 سوالات متداول (FAQ)

<details>
<summary><b>آیا این کد رایگان است؟</b></summary>

بله، برای استفاده شخصی و آموزشی کاملاً رایگان است. برای استفاده تجاری با من تماس بگیرید.
</details>

<details>
<summary><b>آیا می‌توانم مدل خودم را آموزش دهم؟</b></summary>

این نسخه فقط برای استفاده از مدل‌های از پیش آموزش‌دیده است. نسخه آموزشی در آینده منتشر خواهد شد.
</details>

<details>
<summary><b>چطور می‌توانم کمک کنم؟</b></summary>

می‌توانید:
- باگ گزارش دهید
- ویژگی جدید پیشنهاد دهید
- Pull Request بفرستید
- مستندات را بهبود دهید
</details>

<details>
<summary><b>چه سیستمی نیاز دارم؟</b></summary>

- Python 3.7+
- حداقل 4GB RAM
- اتصال اینترنت (برای دانلود اولیه)
- GPU اختیاری (برای سرعت بیشتر)
</details>

---

<div align="center">

## 💖 ساخته شده با عشق توسط رضا صفری فروشانی

**اگر این پروژه برایتان مفید بود، یک ستاره ⭐ بدهید!**

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)

</div>

---

## 📝 تاریخچه تغییرات

### نسخه 1.0.0 (نوامبر 2024)
- ✨ انتشار اولیه
- ✅ پشتیبانی از تشخیص در تصویر
- ✅ پشتیبانی از تشخیص در ویدیو
- ✅ پشتیبانی از تشخیص زنده
- ✅ فیلتر کردن کلاس‌ها
- ✅ منوی تعاملی فارسی

---

**آخرین به‌روزرسانی**: نوامبر 2024  
**نویسنده**: رضا صفری فروشانی  
**وضعیت**: فعال و در حال توسعه  
