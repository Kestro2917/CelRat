import numpy as np
import pandas as pd
import argparse
import sys

def print_input_format_help():
    """Print detailed information about input file formats"""
    help_text = """
Input File Formats for CelRat Analysis:

1. Expression Data File (-d, --data):
   - Tab-separated text file
   - First column: Gene IDs/names
   - First row: Sample IDs
   - Remaining cells: Expression values
   Example:
   GeneID  Sample1  Sample2  Sample3
   Gene1   10.5     8.2      9.1
   Gene2   5.2      6.1      4.8

2. Cell-Type Signature File (-r, --reg):
   - Tab-separated text file
   - First column: Gene IDs/names (matching Expression Data)
   - First row: Cell-type names
   - Remaining cells: Binary values (0 or 1) indicating cell-type markers
   Example:
   GeneID  T-cells  B-cells  Monocytes
   Gene1   1        0        0
   Gene2   0        1        0

Output Format:
- Tab-separated file containing:
  * Cell-type activity scores
  * Differential activity scores between cell types
  * Normalized enrichment scores
"""
    print(help_text)

def CelRat(data, reg, myoutf, perm=1000, median_norm=True, min_num=10):
    """
    Calculate Cell-Type Activity scores using the CelRat algorithm

    Parameters:
    data: DataFrame - expression data
    reg: DataFrame - cell-type signature matrix
    myoutf: str - output file path
    perm: int - number of permutations
    median_norm: bool - whether to perform median normalization
    min_num: int - minimum number of genes required for a cell-type
    """
    # Quantile normalization
    comxx = data.index.intersection(reg.index)
    if len(comxx) == 0:
        print("Error: No common genes found between expression data and cell-type signatures")
        sys.exit(1)

    data = data.loc[comxx]
    reg = reg.loc[comxx]
    xx = reg.sum()

    if sum(xx >= min_num) <= 0:
        print(f"Error: No cell-type has sufficient genes (minimum required: {min_num})")
        sys.exit(1)

    se = xx[xx >= min_num].index
    reg = reg[se]
    pnum = reg.shape[1]

    print(f"\nProcessing {pnum} cell types with {len(comxx)} genes")

    myrk = np.zeros((data.shape[0], data.shape[1]))
    xx = np.zeros_like(myrk)

    for k in range(data.shape[1]):
        myrk[:, k] = data.iloc[:, k].rank()
        xx[:, k] = np.sort(data.iloc[:, k])

    mymed = np.median(xx, axis=1)
    for k in range(data.shape[1]):
        data.iloc[:, k] = mymed[myrk[:, k].astype(int) - 1]

    if median_norm:
        mymed = data.median(axis=1)
        data = data.subtract(mymed, axis=0)

    cnum = data.shape[1]
    rnum = data.shape[0]
    es = np.zeros((cnum, pnum))
    es = pd.DataFrame(es, columns=reg.columns)

    # Create pairwise combinations for difference scores
    celltype_pairs = []
    for i in range(pnum-1):
        for j in range(i+1, pnum):
            celltype_pairs.append(f"{reg.columns[i]}__VS__{reg.columns[j]}")

    es_dif = np.zeros((cnum, len(celltype_pairs)))
    es_dif = pd.DataFrame(es_dif, columns=celltype_pairs)

    # Calculate common genes between cell types
    xx = pd.DataFrame(0, index=reg.index, columns=celltype_pairs)
    count = 0
    minv = np.zeros(len(celltype_pairs))
    for i in range(pnum-1):
        for j in range(i+1, pnum):
            xx.iloc[:, count] = reg.iloc[:, i] * reg.iloc[:, j]
            minv[count] = min(reg.iloc[:, i].sum(), reg.iloc[:, j].sum())
            count += 1

    comgen_num = xx.sum()
    xx = minv - comgen_num
    subset = xx[xx >= min_num].index

    print("\nCalculating Cell-Type Activity Scores")
    for k in range(cnum):
        print(f"\rProcessing sample {k+1}/{cnum}", end="")
        myorder = data.iloc[:, k].sort_values(ascending=False).index
        cur_exp = data.loc[myorder, data.columns[k]]
        cur_reg = reg.loc[myorder]

        fg1 = np.abs(cur_reg.multiply(cur_exp, axis=0))
        bg1 = np.abs((1 - cur_reg).multiply(cur_exp, axis=0))

        fg1 = fg1.cumsum()
        bg1 = bg1.cumsum()

        for i in range(fg1.shape[1]):
            fg1.iloc[:, i] = fg1.iloc[:, i] / fg1.iloc[-1, i]
            bg1.iloc[:, i] = bg1.iloc[:, i] / bg1.iloc[-1, i]

        xx = fg1 - bg1
        tmp = xx.max()
        pos_es = np.where(tmp > 0, tmp, 0)
        tmp = xx.min()
        neg_es = np.where(tmp < 0, tmp, 0)
        es.iloc[k] = np.where(pos_es > np.abs(neg_es), pos_es, neg_es)

        count = 0
        for i in range(pnum-1):
            for j in range(i+1, pnum):
                xx = fg1.iloc[:, i] - fg1.iloc[:, j]
                maxv = max(xx) if max(xx) > 0 else 0
                minv = min(xx) if min(xx) < 0 else 0
                es_dif.iloc[k, count] = maxv if maxv > abs(minv) else minv
                count += 1

    print("\nPerforming permutation test")
    pm_es = np.zeros((pnum, perm))
    pm_es_dif = np.zeros((len(celltype_pairs), perm))

    for k in range(perm):
        print(f"\rPermutation {k+1}/{perm}", end="")
        se = np.random.randint(0, cnum)
        cur_exp = np.random.permutation(data.iloc[:, se])

        fg1 = np.abs(cur_reg.multiply(cur_exp, axis=0))
        bg1 = np.abs((1 - cur_reg).multiply(cur_exp, axis=0))

        fg1 = fg1.cumsum()
        bg1 = bg1.cumsum()

        for i in range(fg1.shape[1]):
            fg1.iloc[:, i] = fg1.iloc[:, i] / fg1.iloc[-1, i]
            bg1.iloc[:, i] = bg1.iloc[:, i] / bg1.iloc[-1, i]

        xx = fg1 - bg1
        tmp = xx.max()
        pos_es = np.where(tmp > 0, tmp, 0)
        tmp = xx.min()
        neg_es = np.where(tmp < 0, tmp, 0)
        pm_es[:, k] = np.where(pos_es > np.abs(neg_es), pos_es, neg_es)

        count = 0
        for i in range(pnum-1):
            for j in range(i+1, pnum):
                xx = fg1.iloc[:, i] - fg1.iloc[:, j]
                maxv = max(xx) if max(xx) > 0 else 0
                minv = min(xx) if min(xx) < 0 else 0
                pm_es_dif[count, k] = maxv if maxv > abs(minv) else minv
                count += 1

    print("\nNormalizing scores")
    std = np.std(np.abs(pm_es), axis=1)
    for k in range(es.shape[1]):
        es.iloc[:, k] = es.iloc[:, k] / std[k]

    std = np.std(np.abs(pm_es_dif), axis=1)
    for k in range(es_dif.shape[1]):
        es_dif.iloc[:, k] = es_dif.iloc[:, k] / std[k]

    res = pd.concat([es, es_dif[subset]], axis=1)
    res.index = data.columns
    res.to_csv(myoutf, sep='\t')

    print(f"\nResults written to: {myoutf}")
    return {
        'cell_type_activity': es,
        'differential_activity': es_dif[subset]
    }

def main():
    parser = argparse.ArgumentParser(
        description='CelRat: Cell-Type Activity Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--data', required=True,
                      help='Expression data file (tab-separated)')
    parser.add_argument('-r', '--reg', required=True,
                      help='Cell-type signature file (tab-separated)')
    parser.add_argument('-o', '--output', required=True,
                      help='Output file path')
    parser.add_argument('-p', '--perm', type=int, default=1000,
                      help='Number of permutations (default: 1000)')
    parser.add_argument('-m', '--min-genes', type=int, default=10,
                      help='Minimum number of genes required per cell-type (default: 10)')
    parser.add_argument('--no-median-norm', action='store_true',
                      help='Disable median normalization')
    parser.add_argument('--format-help', action='store_true',
                      help='Print detailed information about input file formats')

    args = parser.parse_args()

    if args.format_help:
        print_input_format_help()
        sys.exit(0)

    try:
        data = pd.read_csv(args.data, sep='\t', index_col=0)
        reg = pd.read_csv(args.reg, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error reading input files: {str(e)}")
        sys.exit(1)

    CelRat(data, reg, args.output,
           perm=args.perm,
           median_norm=not args.no_median_norm,
           min_num=args.min_genes)

if __name__ == "__main__":
    main()
