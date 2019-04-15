# variational_inference_DP_mix_HDP_topic_ngram
This is a repo of Python (2.7) programs for variational inference for DP mixture of HDP ngram backoff model (generalized by HDP topic model).
See

	Morita, T. (2018) "Unsupervised Learning of Lexical Subclasses from Phonotactics." MIT Ph.D Thesis.

for details on the model.

## Dependencies

- [anaconda2](https://www.anaconda.com/download/)
- [pytables](https://www.pytables.org/usersguide/installation.html)

Python 3 is not supported.

## Hot to Use It?

1. Prepare your data.
	- .tsv file
	- Data column should have strings of comma-separated symbols (e.g. a,b,c for the string "abc").
2. Move to the `code` directory.
2. Run the following:
```bash
python learning.py PATH/TO/YOUR/DATA
```
with the following options:

- Specifying data.
	- `-k`/`--data_column`
		- Column name for the inputs
		- default='IPA_csv'
- Saving results.
	- `-r`/`--result_path`
		- Path to the directory where you want to save results. (Several subdirectories will be created. See below.)
		- default='../results_debug'
	- `-j`/`--jobid`
		- Job ID #. Used as a part of the path to the directory where results are saved (useful for computing clusters).
		- default=Start date & time (e.g. "18-10-20-12-30-14-551728")
- Model parameters.
	- `-n`/`--ngram`
		- Context length of ngram. Only 2 and longer grams are currently supported (i.e., no support for 1gram).
		- default=3
	- `-S`/`--shape_of_sublex_concentration`
		- Shape parameter of the Gamma prior on the concentration of the sublexicon DP.
		- default=10.0
	- `-R`/`--rate_of_sublex_concentration`
		- Rate (= inverse of scale) parameter of the Gamma prior on the concentration of the sublexicon DP.
		- default=10.0
	- `-c`/`--topic_base_counts`
		- Concentration for top level dirichlet distribution.
		- default=1.0
- Variational inference.
	- `-i`/`--iterations`
		- Maxmum # of iterations
		- default=2500
	- `-T`/`--tolerance`
		- Tolerance level to detect convergence
		- default=0.1
	- `-s`/`--sublex`
		- Max # of sublexica
		- default=10


## Results

The program will create subdirectories `[data_filename]/[job_id]` in the directory specified by the `--result_path` option.
For example, if

- your `--result_path` is `../results_eg`
- your data is `../data/example.tsv`, and
- the `-j` option is `10`,

then the results will be saved in `../results_eg/example/10`.

You'll get four files.
- `SubLexica_assignment.csv`
	- Classification probabilities of words (indexed by "customer_id", following the CRP convention).
- `symbol_coding.csv`
	- Code map from b/w data symbols and their integer id.
- `variational_parameters.h5`
	- Variatrional parameters of the model.
- `VI_DP_ngram.log`
	- Log of the update.
	- The recorded "var_bound" (i.e. ELBO) doesn't include constant terms (for computational efficiency).
	- To get the constant term, run `get_var_bound_constant.py`.



## Data Analyzed So Far

- Japanese words appearing in [BCCWJ](https://pj.ninjal.ac.jp/corpus_center/bccwj/data-files/frequency-list/BCCWJ_frequencylist_suw_ver1_0.zip) ([Morita, 2018](#Morita_thesis); [Morita & O'Donnell, under review](#MoritaODonnell_JP)).
- English words in [CELEX](https://catalog.ldc.upenn.edu/LDC96L14) ([Morita, 2018](#Morita_thesis); Morita & O'Donnell, in prep.).
- Tigrinya words collected by [Dr. Kevin Scannell](http://crubadan.org/languages/ti).

### References
- [Morita, Takashi. 2018. Unspervised Learning of Lexical Subclasses from Phonotactics. Ph.D Thesis. Doctoral dissertation. MIT, Cambridge, MA.](https://dspace.mit.edu/bitstream/handle/1721.1/120612/1088558202-MIT.pdf?sequence=1)<a name="Morita_thesis"></a>
- Morita, Takashi and Timothy J. under review. Statistical Evidence for learnable lexical subclasses in Japanese.<a name="MoritaODonnell_JP"></a>