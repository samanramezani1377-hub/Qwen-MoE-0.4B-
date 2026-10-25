# Qwen MoE Mobile — وضعیت پروژه

این مخزن پژوهشی برای بررسی تبدیل FFNهای متنی مدل Qwen3.5-2B به یک معماری sparse MoE و ارزیابی امکان اجرای محلی روی Android/ARM64 است.

> **وضعیت فعلی:** مدل کامل MoE ساخته نشده است. اسکریپت آموزش فقط checkpoint جداگانه برای FFN هر لایه تولید می‌کند. تا زمانی که مدل سفارشی، بارگذاری مجدد، ارزیابی end-to-end و آزمون دستگاه واقعی موفق نشده‌اند، خروجی را نباید مدل آمادهٔ استقرار نامید.

## اجرای آموزش

آموزش مدل فعلاً در **Google Colab** انجام می‌شود؛ GitHub Actions برای آموزش و همگام‌سازی آزمایش‌ها به‌صورت پیش‌فرض غیرفعال شده است. Workflow و کدهای آن حذف نشده‌اند و با opt-in دستی می‌توان مسیر آزمایشی قدیمی را در آینده دوباره فعال کرد.

- [Notebook آموزش همهٔ FFNها در Colab](notebooks/03_colab_full_layer_distillation.ipynb)
- [اسکریپت تقطیر یک FFN](scripts/distill_real_qwen_ffn.py)
- [اسکریپت تقطیر همهٔ FFNها](scripts/distill_all_qwen_ffn_layers.py)
- [اعتبارسنجی checkpointهای ۲۴ لایه](scripts/validate_checkpoint_set.py)
- [Runtime آزمایشی برای جایگزینی FFNها](scripts/moe_runtime.py)
- [ارزیابی سرتاسری معلم در برابر MoE](scripts/evaluate_full_model.py)
- [تولید متن با overlay آزمایشی](scripts/load_moe_model.py)
- [بسته‌بندی checkpointها با hash و manifest](scripts/package_moe_adapter.py)
- [بنچمارک سرتاسری تولید و حافظه](scripts/benchmark_moe_generation.py)

## معماری آزمایشی فعلی

تنظیم پیش‌فرض اسکریپت تمام‌لایه‌ها: 32 expert کانال‌شاردشده و Top-2، با جداسازی ثابت اسناد train/eval در همهٔ لایه‌ها. این فقط یک روش آزمایشی برای تقطیر FFN است؛ هنوز به‌معنای تبدیل کامل Qwen به MoE نیست.

**هدف تثبیت‌شده:** حدود ۱٫۱B پارامتر فعال، با اولویت حفظ استدلال و کیفیت پاسخ. برآورد فرمولی فعلی حدود ۱٫۰۸۶B است؛ این عدد باید پس از بسته‌بندی با شمارش دقیق تأیید شود. 32/Top-2 تعداد پارامترهای ذخیره‌شدهٔ FFN را کاهش نمی‌دهد؛ فقط تعداد کانال‌های FFN فعال در هر توکن را کاهش می‌دهد.

## اسناد مهم

- [ممیزی عمیق تبدیل MoE و معیارهای پذیرش](docs/MOE_CONVERSION_AUDIT.md)
- [بودجه و فرض‌های معماری](docs/ARCHITECTURE_BUDGET.md)
- [راهنمای تقطیر فعال‌سازی FFN](docs/REAL_ACTIVATION_DISTILLATION.md)
- [طرح ساخت مدل کامل](docs/FULL_MODEL_BUILD_PLAN.md)
- [وضعیت پیاده‌سازی و محدودیت‌های باقی‌مانده](docs/IMPLEMENTATION_STATUS.md)

## معیار آماده‌بودن

ابزارهای اعتبارسنجی checkpoint و ارزیابی end-to-end اضافه شده‌اند؛ اما هنوز اجرای واقعی، کیفیت قابل‌قبول، بسته‌بندی مستقل Transformers و سرعت Android اثبات نشده‌اند. مدل فقط زمانی آماده تلقی می‌شود که همهٔ checkpointها معتبر باشند، بارگذاری و inference سرتاسری موفق شود، معیارهای مستقل کیفیت گزارش شوند و سرعت/حافظه روی Android/ARM64 واقعی اندازه‌گیری شود.
