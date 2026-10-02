# Testy — regresje labu PhiMost

Testy żyją tu (`doswiadczenie/testy/`), silnik zostaje w root labu.
`_labroot.py` / `conftest.py` dokładają root na `sys.path`.

## Uruchomienie

```powershell
cd "C:\Users\drwis\model_Ai_breit"

# pojedynczy plik (python)
python doswiadczenie/testy/test_not_flat.py
python doswiadczenie/testy/test_mask.py

# pytest (Windows: python -m pytest)
python -m pytest doswiadczenie/testy/test_karmazyn_lingua.py
python -m pytest doswiadczenie/testy -q
```
