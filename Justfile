set windows-powershell := true

setup:
    #python3 -m venv my-env
    ./my-env/bin/pip install numpy scipy

[windows]
run:
    my-env\Scripts\activate; python TestCases_kmeans.py

[unix]
run:
    ./my-env/bin/python TestCases_knn.py
