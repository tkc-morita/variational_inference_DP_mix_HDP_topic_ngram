# variational_inference_DP_mix_HDP_topic_ngram
This is a bundle of Python (2.7) programs for variational inference for DP mixture of HDP ngram backoff model (generalized by HDP topic model).
See

	Morita, T. (2018) "Unsupervised Learning of Lexical Subclasses from Phonotactics." MIT Ph.D Thesis.

for details on the model.

## Dependencies

- [anaconda2](https://www.anaconda.com/download/)
- [pytable](https://www.pytables.org/usersguide/installation.html)

Python 3 is not supported.

## Hot to Use?

1. Prepare your data.
	- .tsv file
	- Data column should have strings of comma-separated symbols (e.g. a,b,c for the string "abc").
2. Run the following:
```bash
python learning.py PATH/TO/YOUR/DATA
```
with the following options:

- `-n`/`--ngram`
	- Context length of ngram
	- default=3
- `-i`/`--iterations`
	- Maxmum # of iterations
	- default=2500
- `-T`/`--tolerance`
	- Tolerance level to detect convergence
	- default=0.1
- `-s`/`--sublex`
	- Max # of sublexica
	- default=10
- `-c`/`--topic_base_counts`
	- Concentration for top level dirichlet distribution
	- default=1.0
- `-j`/`--jobid`
	- Job ID #
	- default=Start date & time (e.g. )
	parser.add_argument("-k", "--data_column", type=str, help="Column name for the inputs.", default='IPA_csv')