# گزارش پیشرفت آماده‌سازی آموزش MoE — 2026-10-25

> **وضعیت در زمان ثبت:** آماده‌سازی داده و checkpointهای اولیه تأیید شده؛ آموزش هنوز شروع نشده است. این سند گزارش وضعیت اجرای Colab/Google Drive است، نه ادعای اجرای آموزش یا انتشار مدل نهایی.

## مخزن و نوت‌بوک اصلی

- مخزن: https://github.com/samanramezani1377-hub/Qwen-MoE-0.4B-
- نوت‌بوک مسیر واحد: [notebooks/02_colab_chatgpt_text_qwen_logits.ipynb](../notebooks/02_colab_chatgpt_text_qwen_logits.ipynb)
- استخراج logits معلم: [scripts/capture_qwen_logits_for_chatgpt_data.py](../scripts/capture_qwen_logits_for_chatgpt_data.py)
- اسکریپت آموزش ترکیبی: [scripts/train_moe_chatgpt_qwen_kd.py](../scripts/train_moe_chatgpt_qwen_kd.py)
- ساخت checkpoint اولیه: [scripts/build_moe_initial_checkpoints.py](../scripts/build_moe_initial_checkpoints.py)
- تحلیل کانال‌ها: [scripts/analyze_expert_channels.py](../scripts/analyze_expert_channels.py)
- اعتبارسنجی checkpointها: [scripts/validate_checkpoint_set.py](../scripts/validate_checkpoint_set.py)

## مدل و تنظیمات ثبت‌شده

- مدل معلم/پایه: `Qwen/Qwen3.5-2B`
- revision ثابت: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- روش آموزش برنامه‌ریزی‌شده: `text_cross_entropy + 0.5 * Qwen sparse-logit KD`
- logits معلم: top-k = 64، واژگان = 248,320، temperature = 1.0، حداکثر طول توالی = 256
- معماری checkpointهای FFN: 24 لایه، 32 expert در هر لایه، top-2، عرض expert = 192، اندازهٔ intermediate = 6144
- سخت‌افزار اجرای آماده‌سازی گزارش‌شده: Google Colab، NVIDIA Tesla T4
- خروجی checkpoint اولیه با برچسب `initialized_not_trained` است؛ این checkpointها **آموزش‌دیده نیستند**.

## کارهای تکمیل‌شده و بررسی‌شده

1. آرشیو آموزش متنی روی Google Drive موجود است: 1,140 ردیف در `train_archive.jsonl`.
2. مجموعهٔ holdout مستقل روی Google Drive موجود است: `eval_holdout.jsonl` و `eval_holdout.txt` (در گزارش قبلی 135 نمونه اعلام شده است).
3. طرح تقسیم کانال expertها موجود است: `architecture_analysis_v2/expert_channel_plan.json`.
4. تعداد 24 فایل checkpoint اولیه ساخته و اعتبارسنجی ساختاری آن‌ها با موفقیت گزارش شده؛ تعداد سازگار 24، مشکلات گزارش‌شده صفر.
5. استخراج logits برای 1,140 نمونه موفق شده است؛ 1,138 نمونه در همان اجرای استخراج جدید تولید و 2 نمونهٔ موجود از قبل رد شده‌اند. شاخص نهایی 1,140 ردیف یکتا دارد.
6. اعتبارسنجی همهٔ 1,140 tensor موفق شده: 70,414 توکن پاسخ، طول پاسخ 9 تا 147 توکن، صفر پاسخ بریده‌شده و صفر خطا. بررسی‌ها شامل کلیدها/شکل tensorها، شناسه‌ها، مقادیر finite، سازگاری شناسه‌های توکن هدف، محدودیت طول، واژگان، top-k و تطابق شناسه‌ها با منبع آموزش بوده‌اند.
7. اصلاح محلی extractor برای تشخیص اندازهٔ واژگان از output embeddings و تبدیل خروجی `BatchEncoding` به شناسه‌های توکن پشتیبان‌گیری شده است.
8. فایل وضعیت بازیابی در Google Drive نوشته شده و همهٔ بررسی‌های آن موفق گزارش شده‌اند: `all_checks_passed: true`.

## مسیرهای پایدار Google Drive

ریشهٔ کار: `/content/drive/MyDrive/qwen-moe-chatgpt-logits`

- دادهٔ آموزش: `train_archive.jsonl`
- دادهٔ ارزیابی مستقل: `eval_holdout.jsonl` و `eval_holdout.txt`
- طرح معماری: `architecture_analysis_v2/expert_channel_plan.json`
- manifest و checkpointهای اولیه: `analysis_initialized_checkpoints_v2/initialization_manifest.json` و `analysis_initialized_checkpoints_v2/layer_00_top2_32experts.pt` تا `layer_23_top2_32experts.pt`
- logits و index: `qwen_logits_output_v2/`
- manifest logits: `qwen_logits_output_v2/manifest.json`
- index logits: `qwen_logits_output_v2/index.jsonl`
- نسخهٔ پشتیبان extractor اصلاح‌شده: `recovery_scripts/capture_qwen_logits_for_chatgpt_data.py`
- وضعیت بازیابی: `resume_state.json`
- مسیر برنامه‌ریزی‌شده برای خروجی آموزش: `combined_moe_training_v2/`

هش SHA-256 ثبت‌شده برای extractor پشتیبان:
`3a6604341aef8a315be1309bdcc1009deb8cb4d5cee39e7ed26fbcd6c7dcce3f`

## هنوز انجام نشده

- آموزش ترکیبی هنوز شروع نشده است: `training_started: false`.
- کیفیت مدل آموزش‌دیده، معیارهای holdout و آستانه‌های پذیرش هنوز اندازه‌گیری نشده‌اند.
- ساخت/بارگذاری end-to-end مدل کامل، بسته‌بندی نهایی و benchmark روی دستگاه Android/ARM64 هنوز تأیید نشده‌اند.
- checkpointها و tensorهای حجیم Google Drive در GitHub قرار نگرفته‌اند؛ این سند فقط محل و وضعیت آن‌ها را ثبت می‌کند.
- پشتیبان‌گیری extractor به‌معنی commit شدن آن اصلاح محلی در فایل هم‌نام GitHub نیست؛ برای اطمینان باید diff نسخهٔ Drive و GitHub جداگانه بررسی شود.

## قواعد ادامهٔ کار

1. فقط یک مسیر آموزش اجرا شود: Cross-Entropy پاسخ ChatGPT به‌اضافهٔ 0.5 برابر sparse-logit KD معلم Qwen؛ آزمایش text-only یا logits-only موازی اجرا نشود.
2. پیش از آموزش، اسکریپت trainer و سلول‌های نوت‌بوک از نظر ورودی‌ها، loss، تنظیمات، مسیر خروجی و هرگونه حذف فایل بازبینی شوند.
3. خروجی جدید فقط در `combined_moe_training_v2` نوشته شود.
4. فایل‌های قدیمی در `/content/drive/MyDrive/qwen-moe-full-build/checkpoints` و سایر خروجی‌ها حذف یا بازنویسی نشوند. نوت‌بوک را بدون بازبینی سلول‌های پاک‌سازی، یک‌سره اجرا نکنید.
5. پس از آموزش، holdout مستقل ارزیابی شود. معیارهای پذیرش برنامه‌ریزی‌شده: نسبت perplexity حداکثر 1.15 و توافق توکن بعدی حداقل 0.80؛ تا زمان گزارش نتایج نباید آن‌ها را پاس‌شده دانست.
6. روش KD با top-k معلم و یک bucket برای جرم احتمالاتی باقی واژگان، تقریب sparse است و معادل KL دقیق روی کل واژگان نیست.

## فایل بازیابی

JSON وضعیت کامل ذخیره‌شده در Colab/Drive:
`/content/drive/MyDrive/qwen-moe-chatgpt-logits/resume_state.json`

مقادیر کلیدی آن شامل مدل/revision، تعداد نمونه‌ها، top-k، اندازهٔ واژگان، مسیرها، SHA-256 اسکریپت پشتیبان و وضعیت بررسی‌هاست. این JSON وضعیت فایل‌ها را ثبت می‌کند، نه حافظهٔ زندهٔ runtime یا وضعیت آموزش در حال اجرا.
