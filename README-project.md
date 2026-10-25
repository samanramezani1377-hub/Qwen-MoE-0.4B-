# Qwen3.5-2B → Mobile MoE (هدف کل مدل ≈ 2B)

هدف قطعی پروژه این است که کل مدل نهایی حدود ۲ میلیارد پارامتر یکتا داشته باشد؛ هدف قبلی ۰٫۴B پارامتر کل لغو شده است. مدل باید متن‌محور، فارسی‌فهم، استدلال‌محور و قابل اجرای آفلاین روی اندروید باشد. از ابزارهای رایگان استفاده می‌کنیم و هیچ GPU یا سرویس پولی بدون اجازهٔ صریح اجرا نمی‌شود.

## تعریف پارامترها

عدد 508.6M از ضرب vocab=248,320 در hidden_size=2,048 و مربوط به ماتریس embedding مدل مرجع است. اگر embedding و LM head مشترک باشند، این ماتریس فقط یک بار در شمارش کل پارامترهای یکتا حساب می‌شود؛ اما projection خروجی متراکم برای محاسبهٔ logits روی تمام واژگان، همهٔ این وزن‌ها را در مسیر تولید درگیر می‌کند. بنابراین 508.6M را نباید به‌عنوان کل پارامتر فعال all-in گزارش کرد؛ توجه و FFN نیز هزینه دارند. باید سه عدد جدا گزارش شوند: کل پارامترهای یکتا، فعال‌های بدنهٔ Transformer/MoE، و هزینهٔ LM head/logits.

## طرح پایه برای آزمایش

- مدل مرجع: `Qwen/Qwen3.5-2B`
- 24 لایه، hidden size=2048، حفظ ترتیب واقعی لایه‌های linear/full attention از config
- 8 متخصص برای هر FFN تبدیل‌شده، top-2
- expert intermediate size=768 به‌عنوان نقطهٔ شروع
- هر متخصص SwiGLU تقریباً `3 × 2048 × 768 = 4,718,592` پارامتر
- همهٔ متخصص‌ها در 24 لایه: تقریباً 906M پارامتر
- دو متخصص فعال در هر لایه: تقریباً 226.5M پارامتر فعال FFN، بدون attention، embedding/head و اجزای دیگر

این اعداد تخمین اولیه‌اند، نه معماری نهایی. شمارش واقعی وزن‌های checkpoint باید انجام شود و اندازهٔ متخصص‌ها بر اساس آن تنظیم شود تا کل مدل نزدیک 2B بماند.

## دادهٔ فارسی برای آزمایش

گردآورندهٔ `scripts/collect_persian_corpus.py` به‌جای API ویکی‌پدیا از دیتاست‌های آمادهٔ Hugging Face با streaming استفاده می‌کند: منبع اصلی `shekar-ai/Shiraz` با مجوز ODC-BY و جایگزین `ErfanMoosaviMonazzah/mOSCAR-Persian` با مجوز CC-BY-4.0. ورک‌فلو ۶۴ سند واقعی با حداقل ۱۶۰ نویسه می‌گیرد و manifest شامل منبع، مجوز و شناسه/URL سندها می‌سازد. هیچ fallback ساختگی وجود ندارد؛ اگر هر دو منبع ناموفق باشند، آموزش باید متوقف شود. این حجم کوچک فقط برای آزمایش مسیر فنی است، نه آموزش عمومی یا اثبات بهبود استدلال.

## وضعیت

اتصال GitHub به Hugging Face و انتقال فایل‌های پروژه آزمایش شده است. ممیزی واقعی در GitHub Actions انجام شده است: revision مرجع `15852e8c16360a2fea060d615a32b45270f8a8fc`، مجموع 2,274,069,824 عنصر ذخیره‌شده در 632 تانسور. فقط 76,648 بایت header از فایل 4.55GB خوانده شد؛ وزن‌ها دانلود نشدند. تفکیک نام‌محور اولیه: 1.934B متن، 331.4M vision و 8.39M سایر اجزا؛ این‌ها شمارش عناصر تانسور و گروه‌بندی heuristic هستند، نه شمارش قطعی پارامترهای trainable. گزارش و نتیجه در `docs/ARCHITECTURE_BUDGET.md` ثبت شده است. prototype مصنوعی و تست انتقال وزن واقعی لایهٔ صفر Qwen هر دو روی CPU موفق شدند. برای تست واقعی فقط سه tensor FFN با مجموع 75,497,472 بایت از checkpoint 4.55GB دریافت شد؛ جمع 8 shard خروجی dense را با خطای بیشینهٔ 2.12e-7 بازسازی کرد و بودجهٔ کل FFN حفظ شد. این تست فقط نگاشت ریاضی یک FFN را تأیید می‌کند؛ top-2 نیازمند تقطیر/تنظیم‌دقیق است. مدل نهایی MoE هنوز ساخته یا آموزش داده نشده و سرعت/RAM روی گوشی آزموده نشده است.

## پایش پیشرفت و سنجش سرعت

از اجرای بعدی، لاگ‌های GitHub Actions در مرحله‌های بارگذاری مدل، گرفتن فعال‌سازی‌ها، هر batch آموزش، ارزیابی و benchmark به‌صورت JSON و با زمان سپری‌شده، نرخ گام‌ها و ETA چاپ می‌شوند. فایل `experiments/layer0_persian_top2_experiment.benchmark.json` مقایسهٔ FFN اصلی و MoE آموزش‌دیده را با ورودی و dtype یکسان، در batchهای 1 و تا 32 توکن، ثبت می‌کند. معیارها شامل ms/batch، ms/token، token-vectors/sec و نسبت سرعت هستند؛ سربار router و dispatch در سمت MoE حساب می‌شود.

این معیار فقط سرعت forward یک لایه است؛ به‌عنوان سرعت تولید متن مدل کامل، سرعت روی گوشی اندرویدی یا اثبات کیفیت استدلال تفسیر نمی‌شود. عددهای benchmark هنوز باید پس از اجرای جدید تولید شوند و در حال حاضر نتیجه‌ای برای این benchmark جدید ادعا نمی‌شود.

## مراحل بعدی

1. تثبیت گزارش و بازبینی کلیدهای FFN، اجزای MTP و اجزای قابل حذف برای نسخهٔ متن‌محور.
2. طراحی روش warm-up با جمع همهٔ shardها و برنامهٔ distillation/fine-tuning برای حرکت تدریجی به top-2.
3. تبدیل یک لایهٔ واقعی به MoE و سنجش افت perplexity/کیفیت در برابر dense teacher.
4. گسترش تدریجی فقط در صورت حفظ کیفیت و کنترل تعداد پارامترها.
5. تبدیل INT4/INT8 و benchmark CPU/RAM/سرعت روی اندروید؛ هیچ آموزش یا محاسبهٔ پولی بدون اجازه اجرا نمی‌شود.

تخمین خام حافظهٔ وزن‌های 2B: حدود 4GB در FP16، 2GB در INT8 و 1GB در INT4، پیش از سربار کوانتیزه‌سازی، runtime، activations و cache. جزئیات در `docs/ARCHITECTURE_BUDGET.md` و `docs/DECISIONS.md` است.


## Validation after the first MoE distillation run

The unified workflow splits whole Persian documents before tokenization and runs `scripts/validate_moe_experiment.py` against held-out documents. It records relative/normalized errors, zero-output and random-router baselines, repeated layer timing, process memory, and a measured text-backbone parameter inventory plus a clearly labelled all-layer MoE estimate. The 0.4B target is reported separately for transformer-core active parameters and the full-vocabulary logits projection; it is not assumed to be achievable. See `docs/INDEPENDENT_MOE_VALIDATION.md`.


## Active-parameter budget sweep (in progress)

The unified workflow now runs `scripts/sweep_moe_budgets.py` after the document-level Persian split. It compares 8, 16, 32 and 48 channel-sharded experts with top-2 routing, using the same train documents and held-out documents for every variant. To keep the free CPU experiment bounded, it caps training/evaluation token vectors and trains only the real layer-0 FFN; it loads the base checkpoint once per sweep. The report is `experiments/active_budget_sweep.json`.

The sweep reports actual parameter counts from the loaded text backbone, a hypothetical all-layer active-core estimate, the full logits projection separately, held-out FFN-output MSE/relative-L2/cosine similarity, and single-layer throughput. These metrics compare imitation of one dense FFN, **not** full language generation, reasoning, or Android performance. The 450M/400M targets are not assumed reachable by increasing expert count alone: the measured non-FFN core is a lower bound under this particular design. Architecture changes beyond FFN sharding may be required if that floor exceeds a target.
