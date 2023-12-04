# mocap-wrangle-rs

## Description

Code to read a .tsv file with sensor data and output it again to stdout.
Outputs data which is compatible with pandas.
Calculates distance, velocity, acceleration and direction between the 3D points.

Output is used to train NNs on gesture recognition.

### Expected Input

The data generated buy the MoCap system, tab-separated values. This file typically contains a header specifying a number of keywords and
values, such as the number of sensors, framerate and marker names. The program will try to parse this header. Before the first line of the numeric data there is usually a single line with column headers.

The following shows an example of a file header.
```
NO_OF_FRAMES	94577
NO_OF_CAMERAS	14
NO_OF_MARKERS	14
FREQUENCY	200
NO_OF_ANALOG	0
ANALOG_FREQUENCY	0
MARKER_NAMES	RShoulderTop	RElbowOut
TRAJECTORY_TYPES	Measured	Measured
```

The data is expected to be tab-separated values, three for each marker, with an optional frame number and timestamp before the X, Y and Z coordinates.
```
1	0.00000	786.729	-34.016	1064.381 
```

Lines which do not contain the correct number of values are skipped. A warning will be printed to stderr in that case.

To skip a number of lines (of data), use the `--skip` option.

It is also possible to read every N-th line with the `--framestep` option. The default is 1, meaning read every line.

### Output

The output contains a single header line with column names befor the data. Without specifying any other parameters than the filename, the program will output 11 values for each marker. It will calculate the distance, speed, acceleration, azimuth and inclination between each 3D point for each marker in the data. Some calculated values (except for the azimuth and inclination) will be output normalised and standardised as well. In order, the output contains azimuth, inclination, distance, normalised distance, standardised distance, velocity, normalised velocity, standardised velocity, and acceleration, plus normalised and standardised acceleration.

Specifying `--coords` outputs the X, Y and Z coordinate before the calculated values for each marker.
Specifying `--coordsonly`  only outputs the original X, Y and Z coordinates.

Specifying `--timestamp` prefixes each line with a frame number and timestamp (in milliseconds). These are taken from the original data. New frame numbers and timestamps can be generated with the following two parameters.

 - `--startframe` generate new frame number starting at the supplied value.
 - `--starttime` generate new timestamps starting at the supplied value.
 
The timestamp values will be incremented according to the `FREQUENCY` value from the header.

The `--noheader` option skips the header in the output.

The options `--outputstarttimestamp` and `--outputstartframe` skip output until the specified value has been reached.

### Example runs

```
cargo run --release -- -f gestures_ML_05.tsv --timestamp -s2 >gestures_ML_05_data.tsv
cargo run -- -f gestures_ML_05.tsv --coords -k 0
cargo run -- -f gestures_ML_05.tsv --coords --timestamp -k 0
cargo run -- -f gestures_ML_05.tsv  --coordsonly  --noheader
```

Before outputting the data, the program prints what it is doing to stderr.
```shell
13:51:15 [INFO] Args { file: "gestures_ML_05.tsv", verbose: false, coords: false, coordsonly: true, skip: 0, framestep: 1, noheader: true, timestamp: false, starttime: 0, startframe: 0, outputstarttimestamp: 0, outputstartframe: 0, metric: false, keep: None, info: false, nonans: false }
13:51:15 [INFO] Reading header file gestures_ML_05.tsv
13:51:15 [INFO] 00: RShoulderTop
13:51:15 [INFO] 01: RElbowOut
13:51:15 [INFO] 02: RHandIn
13:51:15 [INFO] 03: RHandOut
13:51:15 [INFO] 04: RHip
13:51:15 [INFO] 05: RKnee
13:51:15 [INFO] 06: RToeTip
13:51:15 [INFO] 07: LShoulderTop
13:51:15 [INFO] 08: LElbowOut
13:51:15 [INFO] 09: LHandIn
13:51:15 [INFO] 10: LHandOut
13:51:15 [INFO] 11: LHip
13:51:15 [INFO] 12: LKnee
13:51:15 [INFO] 13: LToeTip
13:51:15 [INFO] Header contains 14 lines, 8 matched.
13:51:15 [INFO] Expecting 94577 frames.
13:51:15 [INFO] Reading file gestures_ML_05.tsv into memory
13:51:15 [INFO] Skipping 14 header lines.
13:51:16 [INFO] Frames 94577, capacity 94577
13:51:16 [INFO] Ready, frames: 94577 (in 956 ms, 98929 l/s)
13:51:16 [INFO] Calculating distances.
13:51:16 [INFO] Calculating velocities.
13:51:16 [INFO] Calculating accelerations.
13:51:16 [INFO] Calculating angles.
13:51:17 [INFO] Calculating min/max.
13:51:17 [INFO] Outputting data.
13:51:17 [WARN] Inclination NaN fixed in frame 11875, sensor 13/LToeTip.
13:51:17 [WARN] Inclination NaN fixed in frame 20443, sensor 4/RHip.
13:51:17 [WARN] Inclination NaN fixed in frame 42111, sensor 13/LToeTip.
13:51:18 [WARN] Inclination NaN fixed in frame 81215, sensor 11/LHip.
13:51:18 [WARN] Inclination NaN fixed in frame 92446, sensor 4/RHip.
13:51:18 [WARN] Inclination NaN fixed in frame 93647, sensor 13/LToeTip.
13:51:18 [INFO] Ready, frames: 94577 (in 1037 ms, 91202 l/s).
```

Without specifying more than the filename, the program will output 

## Source

The source code is available on [HumLab's github](https://github.com/HumLabLu/mocap-wrangle-rs).

## Program Options

```shell
Usage: mocap-wrangle [OPTIONS]

Options:
  -f, --file <FILE>
          [default: street_adapt_1.tsv]
  -v, --verbose
          Produce superfluous output.
  -c, --coords
          Include X, Y and Z coordinates in output.
      --coordsonly
          Only X, Y and Z coordinates in output.
  -s, --skip <SKIP>
          Skip first n columns in sensor data. [default: 0]
      --framestep <FRAMESTEP>
          Read frames in steps. [default: 1]
      --noheader
          Do not output header row.
      --timestamp
          Add frame number and timestamp.
      --starttime <STARTTIME>
          Starting time (ms). [default: 0]
      --startframe <STARTFRAME>
          Starting frame number. [default: 0]
      --outputstarttimestamp <OUTPUTSTARTTIMESTAMP>
          Start outputting at this timestamp (ms). [default: 0]
      --outputstartframe <OUTPUTSTARTFRAME>
          Start outputting at this frame number. [default: 0]
  -m, --metric
          Convert vel/acc to m/s.
  -k, --keep [<KEEP>...]
          Sensor IDs
  -i, --info
          Only parse header and print info.
  -n, --nonans
          Do not emit warnings for NaNs.
  -h, --help
          Print help
```


## Bugs

 - The data is read into memory, this could be a problem for very large files.
 - When specifying `--coords` the aS is not output.
 - Specifying coordsonlys still calculates the other values internally.

## Disclaimer

Released as is.
