# laba_3_IS — ЛР3 по поиску в RAG

Готовый отчёт — [report.pdf](report.pdf). В [results](results) сохранены выдачи всех методов, 600 новых ответов, итоговые метрики и два графика. Показ готовых результатов работает отдельно от ЛР2.

В PowerShell с Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe show_results.py --mlflow
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5002
```

Откройте http://127.0.0.1:5002 и эксперимент lr3.

Для воспроизведения рядом с lab3 нужен клон isachvl/2_Laba_IS в папке lab2. В lab2 создайте .venv, установите PyTorch CPU и requirements.txt. Для новых ответов выполните download_runtime.py в lab2. В lab3 запустите run.ps1. Готовые ответы пропускаются. Чтобы пересчитать отчёт по сохранённым данным, запустите finalize.py через Python из окружения lab2.
