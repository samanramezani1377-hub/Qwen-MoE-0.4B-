# Qwen MoE Mobile — وضعیت پروژه

این مخزن پژوهشی برای بررسی تبدیل FFNهای متنی مدل Qwen3.5-2B به یک معماری sparse MoE و ارزیابی امکان اجرای محلی روی Android/ARM64 است.

> **وضعیت فعلی:** مدل کامل MoE ساخته نشده است. اسکریپت آموزش فقط checkpoint جداگانه برای FFN هر لایه تولید می‌کند. تا زمانی که مدل سفارشی، بارگذاری مجدد، ارزیابی end-to-end و آزمون دستگاه واقعی موفق نشده‌اند، خروجی را نباید مدل آمادهٔ استقرار نامید.

## اجرای آموزش

**مسیر اصلی و واحد آموزش متنی فعلی** در Google Colab است: [نوت‌بوک آموزش ترکیبی متن ChatGPT + logits معلم Qwen](notebooks/02_colab_chatgpt_text_qwen_logits.ipynb). این مسیر پاسخ‌های متنی را با Cross-Entropy و سیگنال logits معلم Qwen را با sparse KD در یک حلقهٔ آموزش استفاده می‌کند؛ مسیرهای text-only و logits-only اجرا نمی‌شوند. آرشیو متنی این نسخه از `data/chatgpt_prompt_response_archive_part2.jsonl` در GitHub خوانده می‌شود؛ خروجی‌های حجیم logits و آموزش و checkpointها همچنان روی Google Drive می‌مانند. GitHub Actions آموزش مدل را اجرا نمی‌کند.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samanramezani1377-hub/Qwen-MoE-0.4B-/blob/main/notebooks/02_colab_chatgpt_text_qwen_logits.ipynb)

- [نوت‌بوک آموزش ترکیبی واحد](notebooks/02_colab_chatgpt_text_qwen_logits.ipynb)
- [اسکریپت استخراج logits هم‌تراز با پاسخ](scripts/capture_qwen_logits_for_chatgpt_data.py)
- [اسکریپت آموزش ترکیبی MoE با CE + sparse KD](scripts/train_moe_chatgpt_qwen_kd.py)
- [پرامپت تولید دادهٔ متنی ChatGPT](docs/CHATGPT_TRAINING_DATA_PROMPT.md)
- [راهنمای آرشیو دیتاست](data/README.md)
- [اعتبارسنجی JSONL دیتاست](scripts/validate_chatgpt_archive.py)
- [اعتبارسنجی checkpointهای ۲۴ لایه](scripts/validate_checkpoint_set.py)
- [Runtime آزمایشی برای جایگزینی FFNها](scripts/moe_runtime.py)
- [ارزیابی سرتاسری معلم در برابر MoE](scripts/evaluate_full_model.py)
- [تولید متن با overlay آزمایشی](scripts/load_moe_model.py)
- [بسته‌بندی checkpointها با hash و manifest](scripts/package_moe_adapter.py)
- [بنچمارک سرتاسری تولید و حافظه](scripts/benchmark_moe_generation.py)

نوت‌بوک‌های \`01\` و \`03\` و اسکریپت‌های activation-distillation ابزارهای پژوهشی قبلی هستند و توسط مسیر اصلی ترکیبی اجرا نمی‌شوند.

## معماری: تحلیل قبل از ساخت اکسپرت

نوت‌بوک اصلی ابتدا وزن‌ها و فعال‌سازی‌های واقعی FFN را تحلیل می‌کند و برای هر لایه گروه‌های متوازن کانال می‌سازد؛ سپس checkpointهای اولیه را از همان وزن‌های pretrained تولید می‌کند. این مرحله آموزش جداگانه نیست. تنها آموزش، متن ChatGPT + تقطیر sparse logits معلم Qwen در یک حلقه است. جزئیات: [طرح تحلیل‌محور](docs/ANALYSIS_FIRST_MOE_PLAN.md), [تحلیل کانال‌ها](scripts/analyze_expert_channels.py), [ساخت checkpoint اولیه](scripts/build_moe_initial_checkpoints.py).

با 32 expert، top-2 و عرض هر expert برابر 192، پارامتر فعال FFN تقریباً 56.6M است؛ اما attention و projection خروجی واژگان متراکم باقی می‌مانند و کل مسیر فعال حدود 1.09B برآورد می‌شود. بنابراین سقف 86M فقط برای فعال‌های FFN قابل اعمال است، نه کل مدل all-in. MoE نیز وزن‌های ذخیره‌شده را خودکار کوچک نمی‌کند.

## گزارش پیشرفت آزمایش جاری

- [گزارش آماده‌سازی و وضعیت بازیابی تا 2026-10-25](docs/EXPERIMENT_PROGRESS_2026-10-25.md) — تعداد داده‌ها، اعتبارسنجی logits، checkpointهای اولیه، مسیرهای Google Drive، SHA-256 پشتیبان و کارهای باقی‌مانده. آموزش هنوز شروع نشده و فایل‌های حجیم روی Google Drive هستند.

## اسناد مهم

- [پرامپت آمادهٔ تولید دادهٔ متنی ChatGPT](docs/CHATGPT_TRAINING_DATA_PROMPT.md)
- [ممیزی عمیق تبدیل MoE و معیارهای پذیرش](docs/MOE_CONVERSION_AUDIT.md)
- [بودجه و فرض‌های معماری](docs/ARCHITECTURE_BUDGET.md)
- [طرح تحلیل‌محور اکسپرت‌ها و تعریف دقیق پارامتر فعال](docs/ANALYSIS_FIRST_MOE_PLAN.md)
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

آرشیو متنی نسخهٔ فعلی در GitHub نگهداری می‌شود: [`data/chatgpt_prompt_response_archive_part2.jsonl`](data/chatgpt_prompt_response_archive_part2.jsonl) (۱٬۲۷۵ نمونه؛ تعداد نهایی را با validator بازبینی کن). اسکریپت `scripts/validate_chatgpt_archive.py` ساختار JSONL، یکتایی شناسه‌ها و نبود رکورد خالی را بررسی می‌کند؛ این بررسی در GitHub Actions نیز اجرا می‌شود. نوت‌بوک دیگر برای ورودی متن به `chatgpt_prompt_response_archive.jsonl` روی Drive مراجعه نمی‌کند. خروجی‌های اجرای جدید با پسوند `_v2` در `MyDrive/qwen-moe-chatgpt-logits/` ذخیره می‌شوند تا با logits/checkpointهای قبلی اشتباه نشوند. نوت‌بوک فقط فایل‌های قدیمی با الگوی دقیق `layer_00_top2_32experts.pt` تا `layer_23_top2_32experts.pt` را از پوشهٔ قدیمی `MyDrive/qwen-moe-full-build/checkpoints/` پاک می‌کند؛ وزن‌های پایه، logits، وضعیت optimizer و فایل‌های دیگر دست‌نخورده می‌مانند. این پاک‌سازی در نشست فعلی اجرا/تأیید نشده است. نوت‌بوک `notebooks/02_colab_chatgpt_text_qwen_logits.ipynb` یک مسیر واحدِ text CE + sparse KD دارد؛ مقایسهٔ text-only/logits-only یا آزمایش‌های جایگزین اجرا نمی‌شوند. workflow اعتبارسنجی ایستا اضافه شده، اما نتیجهٔ اجرای آن در GitHub هنوز تأیید نشده است.

## اجرای مسیر ترکیبی روی Kaggle GPU

اتصال Kaggle API از GitHub Actions تأیید شده است. نوت‌بوک مخصوص Kaggle در مسیر kaggle/combined_moe/ قرار دارد و خروجی‌ها را در /kaggle/working نگه می‌دارد؛ هیچ فایل قدیمی Google Drive را حذف نمی‌کند.

1. به تب [Actions](https://github.com/samanramezani1377-hub/Qwen-MoE-0.4B-/actions/workflows/kaggle-combined-moe.yml) برو و **Run workflow** را بزن.
2. نیازی به واردکردن نام کاربری نیست؛ workflow از قبل روی `samanramezani` تنظیم شده است.
3. اجرای workflow نوت‌بوک را آپلود و اجرای Kaggle را شروع می‌کند؛ آموزش روی GPU Kaggle انجام می‌شود و به سهمیه/دسترس‌پذیری GPU وابسته است.
4. خروجی‌های آموزش و ارزیابی در صفحهٔ Kernel در Kaggle، تب **Output** قابل دریافت‌اند.

نوت‌بوک خصوصی تنظیم شده است. وزن‌های پایه از Hugging Face دریافت می‌شوند. GitHub Actions خودش GPU آموزش فراهم نمی‌کند.


### همگام‌سازی نتیجهٔ Kaggle با GitHub

پس از پایان اجرای Kaggle، workflow فایل گزارش فشردهٔ `qwen_moe_results.json` را از خروجی Kernel می‌خواند و گزارش را در `results/kaggle/latest/result.json` commit می‌کند. این workflow هر ۱۵ دقیقه وضعیت Kernel را بررسی می‌کند؛ می‌توان آن را هم از تب Actions به‌صورت دستی اجرا کرد.

- [Workflow همگام‌سازی نتایج](https://github.com/samanramezani1377-hub/Qwen-MoE-0.4B-/actions/workflows/sync-kaggle-moe-results.yml)
- [گزارش آخرین اجرای Kaggle در مخزن](results/kaggle/latest/result.json)

فایل گزارش شامل وضعیت اجرا، تعداد نمونه‌ها، خلاصهٔ آموزش، اعتبارسنجی checkpointها و معیارهای holdout است. وزن‌های مدل و checkpointهای بزرگ در این گزارش commit نمی‌شوند؛ GitHub فقط گزارش فشرده را دریافت می‌کند. اگر اجرای Kaggle پیش از مرحلهٔ ارزیابی شکست بخورد، ممکن است گزارش معیارهای آموزش در خروجی موجود نباشد؛ وضعیت Kernel در گزارش همگام‌سازی ثبت می‌شود.


## وضعیت و checkpoint پژوهشی

- [گزارش وضعیت و بنچمارک Tesla T4 (2026-10-26)](docs/checkpoints/2026-10-26-benchmark-and-state.md)
- [نتایج بنچمارک به‌صورت JSON](docs/checkpoints/2026-10-26-benchmark-results.json)
- [Google Drive snapshot pointer (2026-10-26)](docs/checkpoints/2026-10-26-drive-snapshot-pointer.md)

این فایل‌ها وضعیت آزمایش را ثبت می‌کنند؛ وزن‌ها و checkpointهای حجیم همچنان در Google Drive نگه‌داری می‌شوند و آموزش اصلی هنوز اجرا نشده است.
