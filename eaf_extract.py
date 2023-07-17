import pandas as pd
import math
import sys, os
import matplotlib.pyplot as mp
import matplotlib as mpl
import matplotlib.dates as dates
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
import argparse
import pympi
import re

# Use PYVENV in ~/Development

# ============================================================================
# Extract targets form the tiers. The classes will be tiername-annotation.
# No annotation will give class 0.
#
# The data file needs to contain the Timestamp field to be able to match
# the EAF data with the MoCap data. Use the --timestamp option if you
# are using the Rust code to process the MoCap data.
#
# (PYVENV) pberck@ip21-178 mocap %
#   python eaf_extract.py -e gestures_ML_05.eaf -F LHandIn_in -t LHand
# or
#   python eaf_extract.py -e gestures_ML_05.eaf -F ".HandIn_*" -t LHand
# or
#   python eaf_extract.py -e gestures_ML_05.eaf -F ".HandIn_.$" -t LHand -t RHand
#
# Useable data can be generated as follow.
# cargo run -- -f gestures_ML_05.tsv --timestamp -s2 > gestures_ML_05_data.tsv
# ============================================================================

'''
Workflow:
    generate distances/velocities/etc
    cargo run --release -- -f ~/Downloads/gestures_ML_05.tsv --timestamp  -s2  >gestures_ML_05_data.tsv

    create datafile for NN training
    python eaf_extract.py -d gestures_ML_05_data.tsv -e gestures_ML_05.eaf -F "LHandIn" 
        -t LHand -o gestures_ML_05_data_targets_LHLH_ND.tsv -N -D

    train
    python torch_mocap_12.py -t gestures_ML_05_data_targets_LHLH_ND.tsv 
'''

# We need an output sensor list as well

parser = argparse.ArgumentParser()
parser.add_argument( "-d", "--datafilename", default="gestures_ML_05_data.tsv",
                     help="Data file to use." )
parser.add_argument( "-e", "--eaffilename", default=None,
                     help="EAF file to use." )
parser.add_argument( "-t", "--tiernames", default=[], action="append",
                     help="Tiernames to include in processing." )
parser.add_argument( "-o", "--output", default="eaf_targets.tsv",
                     help="Output filename." )
parser.add_argument( "-F", "--filter", action="append",
                     help="Regexp to filter output sensor names.", default=[] )
parser.add_argument( "-S", "--standardised",  help="Output _S values", action="store_true" ) # standardised
parser.add_argument( "-N", "--normalised", help="Output _N values", action="store_true" ) # normalised
parser.add_argument( "-D", "--distances", help="Output distance values", action="store_true" )
parser.add_argument( "-V", "--velocities", help="Output velocity values", action="store_true" )
parser.add_argument( "-A", "--accelerations", help="Output acceleration values", action="store_true" )
parser.add_argument( "-Z", "--directions", help="Output direction values", action="store_true" ) 
args = parser.parse_args()

# ============================================================================
# Check files.
# ============================================================================

if os.path.exists( args.output ):
    print( f"Output file {args.output} already exists, overwriting." )
    #sys.exit(3)

# ============================================================================
# Read the EAF file.
# ============================================================================

eaf = None
if args.eaffilename:
    if os.path.exists( args.eaffilename ):
        eaf = pympi.Elan.Eaf(file_path=args.eaffilename, author='eaf_extract.py')

# ============================================================================
# Get data.
# ============================================================================

if not os.path.exists( args.datafilename ):
    print( f"Data file {args.datafilename} does not exist, quitting." )
    sys.exit(1)
df_data = pd.read_csv(
    args.datafilename,
    sep="\t"
)

# ============================================================================
# Keep the ones in the filter.
# Assume we have Frame and Timestamp (these are needed, use the --timestamp
# option when generating data with mocap/main.rs).
# ============================================================================

filtered_columns = []
args.filter.append( "Timestamp" ) # These are automaticall filtered.
args.filter.append( "Frame" )
for sensor in df_data.columns: # This is an "or" filter.
    for filter_re in args.filter:
        if re.search( filter_re, sensor ):
            filtered_columns.append( sensor )
if len(filtered_columns) == 2: # If none (only TS and F), take all!
    filtered_columns = df_data.columns
# Check for -S, -N etc
new_cols = []
new_cols.append( "Frame" )
new_cols.append( "Timestamp" )
for col in filtered_columns:
    suffix = col[-2:]
    if args.directions:
        if suffix == "az":
            new_cols.append( col )
        elif suffix == "in":
            new_cols.append( col )
    # If we specify -D and not -N, we get the raw distance,
    # If we specify -D and -N, we only get the normalised distance.
    # The -S adds the standardised distance to one of the above.
    if args.distances:
        if not args.normalised and suffix == "_d":
            new_cols.append( col )
        if suffix == "dN" and args.normalised:
            new_cols.append( col )
        elif suffix == "dS" and args.standardised:
                new_cols.append( col )
    if args.velocities:
        if not args.normalised and suffix == "_v":
            new_cols.append( col )
        if suffix == "vN" and args.normalised:
            new_cols.append( col )
        elif suffix == "vS" and args.standardised:
            new_cols.append( col )
    if args.accelerations:
        if not args.normalised and suffix == "_a":
            new_cols.append( col )
        if suffix == "aN" and args.normalised:
            new_cols.append( col )
        elif suffix == "aS" and args.standardised:
            new_cols.append( col )
df_data = df_data[new_cols]

# ============================================================================
# Print.
# ============================================================================

print( df_data.head() )
print( df_data.tail() )

# ============================================================================
# Get EAF info/tiers/annotations.
# ============================================================================

# Insert the EAF columns.
# Assume we have a Timestamp column.
if "Timestamp" not in df_data.columns:
    print( "Data does not have a timestamp." ) # We could add it...
    sys.exit(4)
time_delta = df_data.loc[df_data.index[1], 'Timestamp'] - df_data.loc[df_data.index[0], 'Timestamp']
print( time_delta, 1.0/time_delta )
# We could add the full text EAF annotation names as well...
#df_data.insert( len(df_data.columns), "EAF", 0 )
#print( df_data.head() )

# get_full_time_interval()
# get_tier_names()

tier_names = []
if eaf:
    tier_names = eaf.get_tier_names()
    print( tier_names )

# Initialising classes here gives a unique class index to
# every annotation across all tiers.
classes = ["NONE"]

# If we did not specify any tiers, we take them all.
if not args.tiernames:
    args.tiernames = tier_names # These can still be null if no EAF file.

#df_targets = df_data.iloc[:, [0, 1]].copy()

for tier in args.tiernames:
    print( tier )
    ####classes = ["NONE"] # Initialising classes here repeats class indices for each tier.
    df_data.insert( len(df_data.columns), tier, 0 ) # tier as "EAF"
    ##df_targets.insert( len(df_targets.columns), tier, 0 ) # tier as "EAF"
    annotation_data = []
    if eaf:
        annotation_data = eaf.get_annotation_data_for_tier( tier )
    #print( annotation_data )
    for a,b,x in annotation_data:
        cl_name = tier + "-" + x
        if cl_name not in classes:
            classes.append( cl_name )
            print( cl_name )
    t_annotations = 0 # time with annotations
    for t0, t1, cl in annotation_data:
        t0m = t0 / 1000
        t1m = t1 / 1000
        cl_name = tier + "-" + cl
        cli = classes.index( cl_name )
        t_delta = t1 - t0
        t_annotations += t_delta

        #                                      percentage of data with annotations 
        print( t0, t0m, t1, t1m, cl_name, cli, "{:.1f}".format(t_annotations*100.0 / t1) )
        '''
        time ...............    class class index
        35450 35.45 37210 37.21  g1    1
        38410 38.41 39530 39.53  g2    2
        '''
        # Instead of EAF, use tier name?
        #df_data.loc[ (df_data['Timestamp']>=t0m) & (df_data['Timestamp']<t1m), 'EAF' ] = cli
        df_data.loc[ (df_data['Timestamp']>=t0m) & (df_data['Timestamp']<t1m), tier ] = cli
        ##df_targets.loc[ (df_data['Timestamp']>=t0m) & (df_data['Timestamp']<t1m), tier ] = cli

print( classes )
print( df_data.head() )
print( df_data.tail() )

pd.set_option('display.max_rows', 500)
print( df_data.loc[ (df_data['Timestamp']>=24.600) & (df_data['Timestamp']<24.620)] )

print( "Saving output in", args.output )
df_data.to_csv(
    args.output,
    index=False,
    sep="\t"
)

sys.exit(0)

print( "-" )
print( df_targets )
print( "-" )

# Expand doesn't really work...
with open(args.output, "w") as f:
    for i in range(0, len(df_data)):
        data_row = df_data.iloc[i, 2:] # 2: to skip frame and TS
        target_row = df_targets.iloc[i]
        for t in range(0, len(args.tiernames)):
            print( data_row.values )
            for v in data_row.values:
                f.write("{}\t".format( v ))
                f.write("{}\n".format(target_row.iloc[t+2]))

print( args )

sys.exit(1)

# This below needs to be in the loop maybe? or have different colums? for each tier
# create new data frame?

# Fill in the time bits... The data between the t0 and t1 timestamps
# get the class index as an EAF target.
for t0, t1, cl in annotation_data:
    t0m = t0 / 1000
    t1m = t1 / 1000
    cli = classes.index( cl ) 
    print( t0, t0m, t1, t1m, cl, cli )
    '''
    time ...............    class class index
    35450 35.45 37210 37.21  g1    1
    38410 38.41 39530 39.53  g2    2
    '''
    df_data.loc[ (df_data['Timestamp']>=t0m) & (df_data['Timestamp']<t1m), 'EAF' ] = cli


