# مسیر ساخت مدل کامل Qwen3.5-2B Mobile MoE

## تصمیم‌های ثابت
- مرجع: `Qwen/Qwen3.5-2B`
- مسیر مدل: متن‌محور، آفلاین و CPU-first برای Android/ARM64
- جایگزینی FFN: 32 متخصص، Top-2 routing در هر لایهٔ متنی
- بودجه: فقط محیط رایگان؛ هیچ GPU یا سرویس پولی بدون اجازهٔ صریح
- checkpoint تک‌لایهٔ موجود فقط یک نمونهٔ فنی است و مدل نهایی محسوب نمی‌شود.

## فاز بعدی: تولید checkpoint برای همهٔ FFNها
اسکریپت `scripts/distill_all_qwen_ffn_layers.py` هر FFN را به‌صورت جداگانه با خروجی FFN متراکم همان لایه تقطیر می‌کند. در هر مرحله:
1. متن‌های آموزشی از فایل UTF-8 خوانده می‌شوند.
2. مدل مرجع متراکم فعال‌سازی‌های ورودی/خروجی FFN همان لایه را تولید می‌کند.
3. وزن‌های FFN به 32 متخصص channel-sharded منتقل می‌شوند.
4. روتر و متخصص‌ها با MSE روی فعال‌سازی‌های واقعی آموزش داده می‌شوند.
5. checkpoint جداگانه و MSE ارزیابی داخلی همان لایه ذخیره می‌شود.
6. مرحله برای لایهٔ بعدی تکرار می‌شود؛ همهٔ فعال‌سازی‌های 24 لایه هم‌زمان در RAM نگه‌داری نمی‌شوند.

نمونهٔ اجرا در محیط رایگان با RAM کافی:
```bash
python scripts/collect_persian_corpus.py --count 2000 --min-chars 160 --max-chars 4000 --max-scan 50000 --output data/fa_corpus.txt --manifest-output data/fa_corpus.manifest.json
python scripts/split_persian_corpus.py --input data/fa_corpus.txt --train-output data/fa_train.txt --eval-output data/fa_eval.txt --manifest-output data/fa_split.manifest.json --eval-fraction 0.05 --seed 20261025
python scripts/distill_all_qwen_ffn_layers.py --text-file data/fa_train.txt --output-dir artifacts/full_moe --num-experts 32 --top-k 2 --epochs 3 --max-length 96 --batch-size 1 --train-batch-size 32 --device auto
```

این فرمان نمونهٔ شروع است، نه ادعای اینکه گردآوری 2000 سند در همهٔ منابع/اجراها تضمین شده است. باید manifest بررسی شود و train/eval documents مستقل بمانند. اگر حافظه یا زمان محدود شد، می‌توان ابتدا بازهٔ لایه‌ها را با `--start-layer` و `--end-layer` ادامه داد؛ برای خروجی کامل باید همهٔ لایه‌ها checkpoint داشته باشند.

## چیزی که هنوز باید ساخته شود
Per-layer checkpoints هنوز برای بارگذاری به‌عنوان مدل کامل کافی نیستند. قدم پیاده‌سازی بعدی باید:
- پیاده‌سازی مدل/کلاس سازگار با Transformers که در همهٔ لایه‌های هدف، FFN dense را با Top-2 MoE جایگزین کند؛
- بارگذاری strict تمام 24 checkpoint و رد کردن هر checkpoint مفقود یا ناسازگار؛
- ذخیرهٔ مدل و config سفارشی قابل بارگذاری مجدد، همراه tokenizer و مستندات مجوز؛
- تست round-trip: ذخیره، بارگذاری مجدد و تطبیق شکل خروجی logits؛
- ارزیابی end-to-end در برابر مدل مرجع روی اسناد فارسی دیده‌نشده، مجموعهٔ استدلالی، طول متن‌های مختلف و promptهای چندمرحله‌ای؛
- سپس کوانتیزه‌سازی و اتصال به runtime موبایل. قالب GGUF/llama.cpp یا هر runtime دیگر نباید تا زمانی که پشتیبانی MoE سفارشی و معماری دقیق Qwen3.5 تأیید نشده، مفروض گرفته شود.

## معیار پایان
فقط وقتی می‌توان گفت «مدل کامل ساخته شده» که هر لایهٔ هدف checkpoint داشته باشد، مدل سفارشی از پوشهٔ خروجی دوباره بارگذاری شود، inference end-to-end موفق شود و گزارش کیفیت/حافظه/سرعت تولید شود. سرعت یک FFN به‌تنهایی معیار سرعت مدل کامل یا گوشی نیست.

## محدودیت آموزشی
این روش تقطیر، خروجی FFN متراکم را هدف می‌گیرد؛ خودش به‌تنهایی آموزش مستقیم روی پاسخ‌های صحیح یا حفظ استدلال را تضمین نمی‌کند. اگر مدل کامل پس از تبدیل افت زیادی نشان دهد، مرحلهٔ بعدی باید fine-tuning در سطح مدل کامل با دادهٔ مناسب باشد، نه صرفاً زیاد کردن epochهای هر FFN.


## ابزارهای سخت‌گیرانهٔ اعتبارسنجی و ارزیابی

قبل از ادعای مدل کامل، از `python scripts/validate_checkpoint_set.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --layers 24 --num-experts 32 --top-k 2` استفاده کنید. سپس با یک فایل مستقل از دادهٔ آموزش (یک سند در هر خط) اجرا کنید: `python scripts/evaluate_full_model.py --checkpoint-dir /content/drive/MyDrive/qwen-moe-full-build/checkpoints_v2 --text-file /content/fa_eval.txt --output /content/full_model_eval.json --device auto`. این ارزیابی مدل معلم و مدل patched را روی همان اسناد مقایسه می‌کند. فایل eval نباید در corpus آموزشی وجود داشته باشد.

`moe_runtime.py` یک runtime آزمایشی داخل فرایند برای ارزیابی است، نه یک معماری Hugging Face مستقل. برای توزیع مستقل هنوز باید ماژول مدل سفارشی را در بستهٔ مدل قرار داد، `config.json`/`auto_map` و بارگذاری مجدد را تست کرد، و سپس مسیر inference موبایل را اعتبارسنجی کرد.
