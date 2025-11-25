"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      مثال استفاده سریع از YOLOv5                          ║
║                      Quick Start Example                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║  نویسنده: رضا صفری فروشانی (Reza Safari Froushani)                      ║
║  ایمیل: safarireza@gmail.com                                              ║
║  گیتهاب: https://github.com/reza123reza                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

این فایل نمونه‌های ساده برای شروع سریع کار با YOLOv5 را نشان می‌دهد.

نحوه استفاده:
    python example_usage.py
"""

from yolov5_object_detection import YOLOv5Detector


def example_1_basic_image_detection():
    """
    مثال ۱: تشخیص ساده در تصویر
    Example 1: Simple image detection
    
    نویسنده: رضا صفری فروشانی
    """
    print("\n" + "═"*80)
    print("مثال ۱: تشخیص ساده در تصویر")
    print("Example 1: Simple Image Detection")
    print("═"*80)
    
    # ساخت تشخیص‌دهنده
    detector = YOLOv5Detector()
    
    # راه‌اندازی محیط (فقط بار اول لازم است)
    detector.setup_environment()
    
    # تشخیص در تصویر
    # توجه: مسیر تصویر خود را وارد کنید
    # detector.detect_image("path/to/your/image.jpg")
    
    print("\n✓ مثال ۱ تمام شد")
    print("برای استفاده واقعی، مسیر تصویر را uncomment کنید")


def example_2_detect_only_persons():
    """
    مثال ۲: تشخیص فقط انسان‌ها
    Example 2: Detect only persons
    
    توسط: رضا صفری فروشانی - safarireza@gmail.com
    """
    print("\n" + "═"*80)
    print("مثال ۲: تشخیص فقط انسان‌ها")
    print("Example 2: Detect Only Persons")
    print("═"*80)
    
    detector = YOLOv5Detector(conf_threshold=0.5)
    detector.setup_environment()
    
    # تشخیص فقط کلاس 0 (person)
    # detector.detect_image("street.jpg", classes=[0])
    
    print("\n✓ مثال ۲ تمام شد")


def example_3_detect_vehicles():
    """
    مثال ۳: تشخیص وسایل نقلیه
    Example 3: Detect vehicles
    
    Copyright © 2024 Reza Safari Froushani
    """
    print("\n" + "═"*80)
    print("مثال ۳: تشخیص وسایل نقلیه")
    print("Example 3: Detect Vehicles")
    print("═"*80)
    
    detector = YOLOv5Detector()
    detector.setup_environment()
    
    # کلاس‌های وسایل نقلیه:
    # 2: car, 3: motorcycle, 5: bus, 6: train, 7: truck
    vehicles = [2, 3, 5, 6, 7]
    
    # detector.detect_image("parking.jpg", classes=vehicles)
    
    print("\n✓ مثال ۳ تمام شد")


def example_4_video_detection():
    """
    مثال ۴: تشخیص در ویدیو
    Example 4: Video detection
    
    نوشته شده توسط: رضا صفری فروشانی
    """
    print("\n" + "═"*80)
    print("مثال ۴: تشخیص در ویدیو")
    print("Example 4: Video Detection")
    print("═"*80)
    
    detector = YOLOv5Detector(
        weights='yolov5m.pt',
        conf_threshold=0.6
    )
    detector.setup_environment()
    
    # تشخیص در ویدیو
    # detector.detect_video("traffic.mp4")
    
    print("\n✓ مثال ۴ تمام شد")


def example_5_webcam_detection():
    """
    مثال ۵: تشخیص زنده از وبکم
    Example 5: Live webcam detection
    
    GitHub: https://github.com/reza123reza
    """
    print("\n" + "═"*80)
    print("مثال ۵: تشخیص زنده از وبکم")
    print("Example 5: Live Webcam Detection")
    print("═"*80)
    
    detector = YOLOv5Detector(conf_threshold=0.6)
    detector.setup_environment()
    
    print("\nبرای شروع، uncomment کنید:")
    print("For starting, uncomment the line below:")
    # detector.detect_webcam()
    
    print("\n✓ مثال ۵ تمام شد")


def example_6_high_accuracy():
    """
    مثال ۶: تشخیص با دقت بالا
    Example 6: High accuracy detection
    
    توسعه یافته توسط: رضا صفری فروشانی
    Email: safarireza@gmail.com
    """
    print("\n" + "═"*80)
    print("مثال ۶: تشخیص با دقت بالا")
    print("Example 6: High Accuracy Detection")
    print("═"*80)
    
    # استفاده از مدل بزرگ‌تر و آستانه بالاتر
    detector = YOLOv5Detector(
        weights='yolov5l.pt',  # مدل بزرگ‌تر
        conf_threshold=0.8      # آستانه بالاتر
    )
    detector.setup_environment()
    
    # detector.detect_image("important_image.jpg")
    
    print("\n✓ مثال ۶ تمام شد")


def example_7_fast_detection():
    """
    مثال ۷: تشخیص سریع
    Example 7: Fast detection
    
    نویسنده: رضا صفری فروشانی - https://github.com/reza123reza
    """
    print("\n" + "═"*80)
    print("مثال ۷: تشخیص سریع")
    print("Example 7: Fast Detection")
    print("═"*80)
    
    # استفاده از مدل کوچک‌تر برای سرعت بیشتر
    detector = YOLOv5Detector(
        weights='yolov5s.pt',  # مدل کوچک و سریع
        conf_threshold=0.5
    )
    detector.setup_environment()
    
    # detector.detect_image("test.jpg")
    
    print("\n✓ مثال ۷ تمام شد")


def example_8_animals_detection():
    """
    مثال ۸: تشخیص حیوانات
    Example 8: Detect animals
    
    Copyright © 2024 Reza Safari Froushani
    """
    print("\n" + "═"*80)
    print("مثال ۸: تشخیص حیوانات")
    print("Example 8: Detect Animals")
    print("═"*80)
    
    detector = YOLOv5Detector()
    detector.setup_environment()
    
    # کلاس‌های حیوانات:
    # 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep
    # 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
    animals = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    # detector.detect_image("zoo.jpg", classes=animals)
    
    print("\n✓ مثال ۸ تمام شد")


def show_all_classes():
    """
    نمایش تمام کلاس‌های قابل تشخیص
    Show all detectable classes
    
    نویسنده: رضا صفری فروشانی
    """
    print("\n" + "═"*80)
    print("نمایش تمام کلاس‌های قابل تشخیص")
    print("Show All Detectable Classes")
    print("═"*80)
    
    detector = YOLOv5Detector()
    detector.print_classes()


def main():
    """
    تابع اصلی
    Main function
    
    این تابع منوی انتخاب مثال‌ها را نمایش می‌دهد.
    
    استفاده:
        python example_usage.py
    
    نویسنده: رضا صفری فروشانی (Reza Safari Froushani)
    ایمیل: safarireza@gmail.com
    گیتهاب: https://github.com/reza123reza
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "مثال‌های استفاده از YOLOv5" + " " * 29 + "║")
    print("║" + " " * 20 + "YOLOv5 Usage Examples" + " " * 36 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
    print("║" + "  Author: Reza Safari Froushani".ljust(77) + " ║")
    print("║" + "  ایمیل/Email: safarireza@gmail.com".ljust(77) + " ║")
    print("║" + "  گیتهاب/GitHub: https://github.com/reza123reza".ljust(77) + " ║")
    print("╚" + "═" * 78 + "╝")
    
    while True:
        print("\n" + "═" * 80)
        print("لطفاً یک مثال را انتخاب کنید / Please choose an example:")
        print("-" * 80)
        print("1.  تشخیص ساده در تصویر / Simple Image Detection")
        print("2.  تشخیص فقط انسان‌ها / Detect Only Persons")
        print("3.  تشخیص وسایل نقلیه / Detect Vehicles")
        print("4.  تشخیص در ویدیو / Video Detection")
        print("5.  تشخیص زنده از وبکم / Live Webcam Detection")
        print("6.  تشخیص با دقت بالا / High Accuracy Detection")
        print("7.  تشخیص سریع / Fast Detection")
        print("8.  تشخیص حیوانات / Detect Animals")
        print("9.  نمایش لیست کلاس‌ها / Show All Classes")
        print("10. اجرای همه مثال‌ها / Run All Examples")
        print("0.  خروج / Exit")
        print("═" * 80)
        
        choice = input("\nانتخاب شما / Your choice: ").strip()
        
        if choice == '1':
            example_1_basic_image_detection()
        elif choice == '2':
            example_2_detect_only_persons()
        elif choice == '3':
            example_3_detect_vehicles()
        elif choice == '4':
            example_4_video_detection()
        elif choice == '5':
            example_5_webcam_detection()
        elif choice == '6':
            example_6_high_accuracy()
        elif choice == '7':
            example_7_fast_detection()
        elif choice == '8':
            example_8_animals_detection()
        elif choice == '9':
            show_all_classes()
        elif choice == '10':
            print("\n" + "═" * 80)
            print("اجرای همه مثال‌ها...")
            print("Running all examples...")
            print("═" * 80)
            example_1_basic_image_detection()
            example_2_detect_only_persons()
            example_3_detect_vehicles()
            example_4_video_detection()
            example_5_webcam_detection()
            example_6_high_accuracy()
            example_7_fast_detection()
            example_8_animals_detection()
            show_all_classes()
            print("\n✓ همه مثال‌ها اجرا شدند")
        elif choice == '0':
            print("\n" + "═" * 80)
            print("خروج از برنامه...")
            print("Exiting...")
            print("\nبا تشکر از استفاده / Thank you for using")
            print("نویسنده / Author: رضا صفری فروشانی / Reza Safari Froushani")
            print("═" * 80)
            break
        else:
            print("\n⚠ انتخاب نامعتبر! / Invalid choice!")
        
        input("\n[Enter] را فشار دهید تا ادامه یابد / Press [Enter] to continue...")


if __name__ == "__main__":
    """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
    ║  حق نشر © ۲۰۲۴ رضا صفری فروشانی. تمامی حقوق محفوظ است.                   ║
    ║                                                                            ║
    ║  ایمیل / Email: safarireza@gmail.com                                      ║
    ║  گیتهاب / GitHub: https://github.com/reza123reza                          ║
    ║                                                                            ║
    ║  این کد تحت حق نشر محفوظ است.                                            ║
    ║  کپی برداری بدون ذکر منبع ممنوع است.                                     ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    main()
