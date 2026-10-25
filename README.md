# Qwen MoE Mobile — وضعیت پروژه

این مخزن پژوهشی برای بررسی تبدیل FFNهای متنی مدل Qwen3.5-2B به یک معماری sparse MoE و ارزیابی امکان اجرای محلی روی Android/ARM64 است.

> **وضعیت فعلی:** مدل کامل MoE ساخته نشده است. اسکریپت آموزش فقط checkpoint جداگانه برای FFN هر لایه تولید می‌کند. تا زمانی که مدل سفارشی، بارگذاری مجدد، ارزیابی end-to-end و آزمون دستگاه واقعی موفق نشده‌اند، خروجی را نباید مدل آمادهٔ استقرار نامید.

## اجرای آموزش

**مسیر اصلی و واحد آموزش متنی فعلی** در Google Colab است: [نوت‌بوک آموزش ترکیبی متن ChatGPT + logits معلم Qwen](notebooks/02_colab_chatgpt_text_qwen_logits.ipynb). این مسیر پاسخ‌های متنی را با Cross-Entropy و سیگنال logits معلم Qwen را با sparse KD در یک حلقهٔ آموزش استفاده می‌کند؛ مسیرهای text-only و logits-only اجرا نمی‌شوند. داده روی Google Drive می‌ماند و GitHub Actions آموزش مدل را اجرا نمی‌کند.

- [نوت‌بوک آموزش ترکیبی واحد](notebooks/02_colab_chatgpt_text_qwen_logits.ipynb)
- [اسکریپت استخراج logits هم‌تراز با پاسخ](scripts/capture_qwen_logits_for_chatgpt_data.py)
- [اسکریپت آموزش ترکیبی MoE با CE + sparse KD](scripts/train_moe_chatgpt_qwen_kd.py)
- [پرامپت تولید دادهٔ متنی ChatGPT](docs/CHATGPT_TRAINING_DATA_PROMPT.md)
- [اعتبارسنجی checkpointهای ۲۴ لایه](scripts/validate_checkpoint_set.py)
- [Runtime آزمایشی برای جایگزینی FFNها](scripts/moe_runtime.py)
- [ارزیابی سرتاسری معلم در برابر MoE](scripts/evaluate_full_model.py)
- [تولید متن با overlay آزمایشی](scripts/load_moe_model.py)
- [بسته‌بندی checkpointها با hash و manifest](scripts/package_moe_adapter.py)
- [بنچمارک سرتاسری تولید و حافظه](scripts/benchmark_moe_generation.py)

نوت‌بوک‌های \`01\` و \`03\` و اسکریپت‌های activation-distillation ابزارهای پژوهشی قبلی هستند و توسط مسیر اصلی ترکیبی اجرا نمی‌شوند.

## معماری آزمایشی فعلی

تنظیم پیش‌فرض اسکریپت تمام‌لایه‌ها: 32 expert کانال‌شاردشده و Top-2، با جداسازی ثابت اسناد train/eval در همهٔ لایه‌ها. این فقط یک روش آزمایشی برای تقطیر FFN است؛ هنوز به‌معنای تبدیل کامل Qwen به MoE نیست.

**هدف تثبیت‌شده:** حدود ۱٫۱B پارامتر فعال، با اولویت حفظ استدلال و کیفیت پاسخ. برآورد فرمولی فعلی حدود ۱٫۰۸۶B است؛ این عدد باید پس از بسته‌بندی با شمارش دقیق تأیید شود. 32/Top-2 تعداد پارامترهای ذخیره‌شدهٔ FFN را کاهش نمی‌دهد؛ فقط تعداد کانال‌های FFN فعال در هر توکن را کاهش می‌دهد.

## اسناد مهم

- [پرامپت آمادهٔ تولید دادهٔ متنی ChatGPT](docs/CHATGPT_TRAINING_DATA_PROMPT.md)
- [ممیزی عمیق تبدیل MoE و معیارهای پذیرش](docs/MOE_CONVERSION_AUDIT.md)
- [بودجه و فرض‌های معماری](docs/ARCHITECTURE_BUDGET.md)
- [راهنمای تقطیر فعال‌سازی FFN](docs/REAL_ACTIVATION_DISTILLATION.md)
- [طرح ساخت مدل کامل](docs/FULL_MODEL_BUILD_PLAN.md)
- [وضعیت پیاده‌سازی و محدودیت‌های باقی‌مانده](docs/IMPLEMENTATION_STATUS.md)

## معیار آماده‌بودن

ابزارهای اعتبارسنجی checkpoint و ارزیابی end-to-end اضافه شده‌اند؛ اما هنوز اجرای واقعی، کیفیت قابل‌قبول، بسته‌بندی مستقل Transformers و سرعت Android اثبات نشده‌اند. مدل فقط زمانی آماده تلقی می‌شود که همهٔ checkpointها معتبر باشند، بارگذاری و inference سرتاسری موفق شود، معیارهای مستقل کیفیت گزارش شوند و سرعت/حافظه روی Android/ARM64 واقعی اندازه‌گیری شود.


## سیاست نگهداری آرشیو معلم

- در Google Drive، فایل `teacher_generated.jsonl` آرشیو مرجع همهٔ پرامپت‌ها و پاسخ‌های کامل‌شده است؛ تولید نمونه‌به‌نمونه به آن append می‌شود تا قطع Colab باعث از دست رفتن پیشرفت قبلی نشود.
- فایل‌های `teacher_train_subset.jsonl` و `teacher_eval_holdout.jsonl` نماهای مشتق‌شده‌اند؛ می‌توان آن‌ها را دوباره تولید کرد، اما جای آرشیو مرجع را نمی‌گیرند.
- صادرکردن train به‌طور پیش‌فرض به ۲٬۰۰۰ نمونه محدود است. مجموعهٔ eval با hash پایدار شناسهٔ نمونه جدا می‌شود و نباید وارد آموزش شود.
- `teacher_archive_manifest.json` هش SHA-256 آرشیو، شمار نمونه‌ها، تعداد توکن‌های ذخیره‌شده و مسیر خروجی‌ها را ثبت می‌کند.
- اگر آخرین خط JSONL به علت قطع runtime ناقص باشد، بایت‌های آن به فایل `.incomplete_tail` قرنطینه می‌شوند؛ خط‌های خراب میانی به‌صورت خودکار حذف نمی‌شوند و نیازمند بررسی دستی‌اند.
- این طراحی هنوز در Colab واقعی اجرا نشده است؛ قبل از تولید طولانی، ابتدا با چند نمونهٔ آزمایشی اعتبارسنجی شود.


## مسیر جدید: دادهٔ متنی ChatGPT + logits معلم Qwen

- [نوت‌بوک Colab: آموزش واحد متن ChatGPT + logits معلم Qwen](notebooks/02_colab_chatgpt_text_qwen_logits.ipynb)
- [اسکریپت استخراج logits توکن‌به‌توکن Qwen](scripts/capture_qwen_logits_for_chatgpt_data.py)
- [اسکریپت آموزش ترکیبی MoE با CE + sparse KD](scripts/train_moe_chatgpt_qwen_kd.py)

این مسیر اکنون یک مسیر واحد برای آموزش ترکیبی دارد: پاسخ‌های متنی ChatGPT برای Cross-Entropy و logits معلم Qwen برای تقطیر sparse در همان حلقهٔ آموزش. اسکریپت `scripts/capture_qwen_logits_for_chatgpt_data.py` ورودی توکن‌شدهٔ دقیق، شناسه‌های توکن هدف، logits برتر و log-normalizer کل واژگان را ذخیره می‌کند. اسکریپت `scripts/train_moe_chatgpt_qwen_kd.py` مدل کامل را با ۲۴ checkpoint اولیهٔ MoE بارگذاری می‌کند، backbone را فریز می‌کند و expertها/routerها را با `text_cross_entropy + kd_weight * sparse_aggregated_tail_KL` آموزش می‌دهد. KL از top-k معلم و یک bucket برای جرم احتمالاتی باقی واژگان استفاده می‌کند؛ تقریب است، نه KL دقیق واژگان کامل.

دادهٔ آموزشی روی Google Drive نگهداری می‌شود، نه GitHub: `MyDrive/qwen-moe-chatgpt-logits/chatgpt_prompt_response_archive.jsonl`. خروجی logits در `qwen_logits_output/` و checkpointهای آموزش در `combined_moe_training/checkpoints/` قرار می‌گیرند. نوت‌بوک `notebooks/02_colab_chatgpt_text_qwen_logits.ipynb` یک مسیر کامل و واحد را اجرا می‌کند؛ مقایسهٔ text-only/logits-only یا آزمایش‌های جایگزین اجرا نمی‌شوند. کد به‌صورت static بررسی می‌شود، اما فقط اجرای موفق واقعی در Colab می‌تواند آموزش را تأیید کند.
