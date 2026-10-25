# وضعیت پیاده‌سازی Qwen MoE (به‌روزرسانی 2026-10-25)

## تصمیم معماری تثبیت‌شده
- مدل مرجع: Qwen/Qwen3.5-2B.
- 24 لایهٔ متن، FFN کانال‌شاردشده به 32 expert با عرض 192 و Top-2.
- هدف تقریبی پارامتر فعال: 1.1B؛ اولویت حفظ استدلال و کیفیت پاسخ.
- آموزش مدل در Google Colab است. آموزش GitHub Actions حذف نشده، اما jobهای آموزش/همگام‌سازی همچنان opt-in دستی می‌خواهند.

## تغییرات ثبت‌شده در ریپو
1. جداسازی سندی مشترک برای train/eval در تمام لایه‌ها حفظ شده است.
2. اسکریپت تمام‌لایه‌ها Router را با یک warm-start مبتنی بر برچسب‌های greedy آموزش می‌دهد: انتخاب اکسپرت‌هایی که طبق کاهش خطای مربعی، خروجی FFN معلم را بهتر تقریب می‌زنند؛ سپس آموزش مشترک MSE و loss توازن انجام می‌شود. این heuristic است، نه تضمین کیفیت.
3. برای هر checkpoint، revision معلم، شمارش استفادهٔ اکسپرت‌ها، اکسپرت‌های بدون استفاده و entropy مسیریابی ذخیره می‌شود.
4. `scripts/validate_checkpoint_set.py` تعداد/shape/metadata هر 24 checkpoint را بررسی می‌کند و مدل/نسخهٔ معلم ناسازگار را رد می‌کند.
5. `scripts/moe_runtime.py` جایگزینی سخت‌گیرانهٔ FFNهای 24 لایه را برای ارزیابی درون‌فرایندی فراهم می‌کند.
6. `scripts/evaluate_full_model.py` perplexity، cross-entropy و توافق توکن بعدی معلم و مدل تبدیل‌شده را روی اسناد مستقل اندازه می‌گیرد.
7. `scripts/load_moe_model.py` برای تست تولید متن، `scripts/package_moe_adapter.py` برای ساخت overlay دارای hash و manifest، و `scripts/benchmark_moe_generation.py` برای اندازه‌گیری latency/throughput و حافظه اضافه شده‌اند.
8. Notebook Colab اکنون train/eval را در سطح سند جدا می‌کند، checkpointها را اعتبارسنجی می‌کند، ارزیابی سرتاسری انجام می‌دهد و بستهٔ overlay را می‌سازد.

## وضعیت‌هایی که هنوز تکمیل یا اثبات نشده‌اند
- آموزش واقعی مدل در Colab در این تغییرات اجرا نشده است؛ بنابراین هیچ نتیجهٔ کیفیتی یا checkpoint کامل جدیدی ادعا نمی‌شود.
- کد runtime و اسکریپت‌های تازه باید در اجرای واقعی Colab و با نسخهٔ Transformers مورد استفاده تست شوند؛ وضعیت GitHub Actions هنگام ثبت این سند در حال اجرا/انتظار بوده است.
- این overlay شامل وزن‌های پایهٔ Qwen نیست؛ برای اجرا باید مدل پایه جداگانه دریافت شود. هنوز یک بستهٔ مستقل Transformers با `AutoModel`/ `auto_map` ساخته نشده است.
- روش dispatch فعلی برای صحت پژوهشی است و همهٔ expertها را بررسی می‌کند؛ سرعت اندروید بهینه نشده است.
- ارزیابی perplexity و توافق توکن، ارزیابی کامل استدلال نیست. مجموعهٔ مستقل استدلال فارسی، آزمون طولانی‌متن، quantization و benchmark واقعی Android/ARM64 هنوز لازم‌اند.
- مدل نهایی نباید تا زمانی که کیفیت مستقل و آزمون موبایل قابل‌قبول نیست، «آمادهٔ انتشار» معرفی شود.

## دستورهای پس از آموزش در Colab
```bash
python scripts/validate_checkpoint_set.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --layers 24 --num-experts 32 --top-k 2
python scripts/evaluate_full_model.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --text-file /content/drive/MyDrive/qwen-moe-full-build/fa_eval.txt --output /content/full_model_eval.json --device auto
python scripts/package_moe_adapter.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --output-dir /content/qwen-moe-overlay --model Qwen/Qwen3.5-2B --layers 24 --num-experts 32 --top-k 2\npython scripts/benchmark_moe_generation.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --new-tokens 64 --repeats 3 --num-experts 32 --top-k 2 --device auto
```
