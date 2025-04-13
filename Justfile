set windows-powershell := true

[unix]
setup:
    python3 -m venv my-env
    ./my-env/bin/pip install numpy scipy

[unix]
run:
    ./my-env/bin/python TestCases_knn.py

[windows]
setup:
    python -m venv my-env
    ./my-env/Scripts/pip.exe install numpy scipy

[windows]
run:
    my-env\Scripts\activate; python TestCases_knn.py

