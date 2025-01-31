# CelRat

# Get help about command line options
python celrat.py --help

# Get detailed information about input formats
python celrat.py --format-help

# Run the analysis
python celrat.py -d expression_data.txt -r cell_type_signatures.txt -o results.txt -p 1000 --min-genes 10

# Run without median normalization
python celrat.py -d expression_data.txt -r cell_type_signatures.txt -o results.txt --no-median-norm
