#### The qramutils Library

This library allows you to find meaningful parameters for a given dataset in order to be stored on QRAM.
With this library you can calculate without writing the code yourself things like: 
- Frobenius norm of a given matrix
- Sparsity
- A parameter for the current best matrix decomposition for QRAM
- Subroutines to create useful plots
- Conditioning number

    libq = qramutils.QramUtils(X, logging_handler=logging)

    logging.info("Matrix dimension {}".format(X.shape))

    sparsity = libq.sparsity()
    logging.info("Sparsity (0=dense 1=empty): {}".format(sparsity))

    frob_norm = libq.frobenius()
    logging.info("The Frobenius norm: {}".format(frob_norm))

    best_p, min_sqrt_p = libq.find_p()
    logging.info("Best p value: {}".format(best_p))

    logging.info("The \\mu value is: {}".format(min(frob_norm, min_sqrt_p)))

    qubits_used = libq.find_qubits()
    logging.info("Qubits needed to index+data register: {} ".format(qubits_used))


How? It is very simple. First package the library with:

    pipenv run python3 setup.py sdist
    cp dist/qramutils-0.1.0.tar.gz where/you/need/it

Then install it wherever you want:

    pipenv install qramutils-0.1.0.tar.gz 

There is some useful code in the example folder.

$ pipenv run python3 examples/mnist_QRAM.py --help
usage: mnist_QRAM.py [-h] [--db DB] [--generateplot] [--analize]
                     [--pca-dim PCADIM] [--polyexp POLYEXP]
                     [--loglevel {DEBUG,INFO}]

Analyze a dataset and model QRAM parameters

optional arguments:
  -h, --help            show this help message and exit
  --db DB               path of the mnist database
  --generateplot        run experiment with various dimension
  --analize             Run all the analysis of the matrix
  --pca-dim PCADIM      pca dimension
  --polyexp POLYEXP     degree of polynomial expansion
  --loglevel {DEBUG,INFO}
                        set log level

This is how it looks like an execution.



pipenv run python3 examples/mnist_QRAM.py --db data --analize --loglevel INFO
04-01 22:23 root INFO     Calculating parameters for default configuration: PCA dim 39, polyexp 2
04-01 22:24 root INFO     Matrix dimension (60000, 819)
04-01 22:24 root INFO     Sparsity (0=dense 1=empty): 0.0
04-01 22:24 root INFO     The Frobenius norm: 4.6413604982930385
/home/scinawa/.local/share/virtualenvs/qram-utils-HYj0laHn/lib/python3.4/site-packages/numpy/linalg/linalg.py:2296: RuntimeWarning: overflow encountered in power
  ret **= (1 / ord)
04-01 22:26 root INFO     best p 0.8501000000000001
04-01 22:26 root INFO     Best p value: 0.8501000000000001
04-01 22:26 root INFO     The \mu value is: 4.6413604982930385
04-01 22:26 root INFO     Qubits needed to index+data register: 26.0

