set windows-powershell := true

[unix]
setup:
    #python3 -m venv my-env
    ./my-env/bin/pip install numpy scipy pillow matplotlib

[windows]
run:
    my-env\Scripts\activate; python my_labeling.py

[unix]
run:
    ./my-env/bin/python my_labeling.py
