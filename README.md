# Credit-scoring

Repository del progetto di Credit Scoring in collaborazione con Banca Intesa San Paolo per ICSC - Centro Nazionale in HPC, Big Data e Quantum Computing, Spoke 10 - Quantum Computing.

Per girare il codice di test, creare un ambiente conda con python e installare [dwave-hybrid](https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/sdk_index.html):

```
conda create -n <nome-env> python=3.13.2
conda activate <nome-env>
pip install dwave-hybrid==0.6.13
pip install pyyaml pandas
```

Il codice sorgente che costruisce la funzione di costo e risolve il problema con il simulatore di dwave si trova nel file ```cost_function.py```, gli iperparametri si settano nello yaml file ```config.yaml```. Per eseguire il codice il comando da girare all'interno dell'ambiente conda è:
```
python cost_function.py config.yaml 
```

Il codice classico che testa il combinatorio di tutte le disposizioni di n aziende in m classi si trova in ```classical_approach.py```. Anche in questo caso per eseguire il codice è necessario passare in input il file di configurazione con gli iperparametri ```config.yaml```:
```
python classical_approach.py config.yaml
```

<img src="ICSC-logo.png" width="1000">
