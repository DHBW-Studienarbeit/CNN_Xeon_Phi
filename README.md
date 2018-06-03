
# CNN train program optimized for MNIST digit recognition on Intel Xeon Phi Coprozessor

## Contents
- Installation
- Notes
- Usage
- Additional Contents


## Installation:

### necessary Tools:
- GNU make or compatible make-program (installed on it-phi)
- Intel C/C++ Compiler icc (installed on it-phi)
- Intel Math Kernel Library (installed on it-phi)

### recommended tools:
- Python-Interpreter for code generation and to run Tensorflow example code (installed on it-phi)
- Tensor Flow (prerequesits installed on it-phi; tensor flow itself is in users home directory; follow instructions for ubuntu here: https://www.tensorflow.org/versions/r1.1/install/)
- Tensor Flow MNIST example(https://www.tensorflow.org/tutorials/layers)
- cog to generate source code (not installed on it-phi; without this, you cannot change network model and batchsize, but still some other parameters like learnrate, verbosity etc. )


## Notes
- The application uses relatives paths to access train and test data. Please invoke the application by invoking `./build/program` from the project root directory. You can also edit configurqation and use absolute paths on demand.
- it is recommended to edit configuration in python scripts and generate c code using cog. You can also edit the file `src/network_executer/application/include/config.h` directly for changes not affecting network model or batchsize, but regard this unsafe and eventually causing undefined behaviour in future versions.
- You need to change an environment variable to get code generation running: PYTHONPATH has to contain the absolute path of `src` this is necessary so that the generation code can find the python module for network description


## Usage:
1. edit network configuration (skip for using existing configuration)
2. generate source code (skip for using existing configuration)
3. build application
4. run application

### edit network configuration
The network model is defined in the file `src/network_descriptor/NetInstance.py`. This file is pretty self-explaining. Also, there is not really a need to change it unless you want to do other things than MNIST data recognition with this program.
The network configuration is defined in the file `src/network_descriptor/Net_Config.py`.
This file defines a dictionary of config keys to control the program behaviour.
Lets look at these config keys more carefully:
- CONFIG_BATCHSIZE: size of a minibatch. This value has a heavy impact on memory usage of the application. works great with values up to 100, but might cause more page faults with higher values
- CONFIG_FLOATTYPE_FLOAT or CONFIG_FLOATTYPE_DOUBLE: make sure you have EXACTLY one of these in your configuration file. Single precision should be enough for these calculations, but feel free to use double if you want. Assign an empty string to this.
- CONFIG_LEARNING_RATE: initial value of the network's learning rate.
- CONFIG_LEARNRATE_REDUCTION: factor to change learning rate when cost increases between two evaluations. should be somewhere between 0.0f and 1.0f. Use 1.0f to use a fixed learnrate.
- CONFIG_REDUCE_CONV_LEARNRATE: if this is defined, convolutional layers will be trained with a smaller learnrate. (recommended) Assign an empty string to this if you want to use it.
- CONFIG_NUM_OF_ITERATIONS: number of train/test iterations before final test
- CONFIG_TRAININGS_PER_ITERATION: number of minibatches used for training in each iteration
- CONFIG_TESTS_PER_ITERATION: number of minibatches used for testing in each iteration
- CONFIG_NUM_DATASETS_PER_FILE: number of laballed MNIST datasets per .csv file
- CONFIG_DIR_TRAIN: directory containing .csv files with training data
- CONFIG_DIR_TEST: directory containing .csv files with testing data
- CONFIG_NUM_TRAINFILES: number of .csv files for training
- CONFIG_NUM_TESTFILES: number of .csv files for testing
- CONFIG_RANDGEN_SEED: seed used for pseudo random number generation
- CONFIG_RANDGEN_MEAN: mean value of generated random weights. Use 0.0f here.
- CONFIG_RANDGEN_RANGESIZE: standard derivation of initial random weights. Use something small, but more than zero here.
- CONFIG_MATH_VECT_PARALLEL: if this is defined, vector mathematical functions will be calculated parallel distributed over CONFIC_VECT_NUM_THREADS threads. This might increase, but also decrease performance depending on the network model. Assign an empty string to this if you want to use it.
- CONFIC_VECT_NUM_THREADS: Threads to be used if vector mathematical functions are executed parallel
- CONFIG_ARRAY_ALIGNMENT: alignment of arrays in memory. This is not measured in bytes, but in sizeof(floattype) units.
- CONFIG_OUTPUT_VERBOSE: if this is defined, the program will produce more output about the training process. Otherwise, it will just print a summary in the end. Assign an empty string to this if you want to use it.
- CONFIG_NUM_FINAL_TESTS: number of minibatches to be used for the final evaluation

### generate source code
Make sure you have read the notes section above before you do this
You can generate source code for the application from the python scripts described in the section above.
To generate sourcecode, execute the following command on a shell from the projects root directory.
The '@' character is known for causing trouble on windows power shell, but it should work on most other shells. (including cmd.exe on Windows systems)
```
cog -r @COG_filelist.txt
```

### build application
There are several make targets described in the makefile at project root.
`make all` will build the application
`make clean` will clean all build files
`make rebuild` will rebuild the application
`make execute` will build the application and execute it when finished. The execution time will be measured with `/usr/bin/time`, the output of the application and of the tool will be written to the file `report.txt`
There is an option in the makefile which will lead to the build of a native application. In Offload mode, the application is faster than in native mode, so this is not recommended.


### run application
If you want to invoke the application from hand, type the following to a shell on the projects root directory:
```
./build/program
```


## Additional contents
- MNIST data can be found in the directories `train` and `test`. They are equal to the data used for all other implementations.
- some example reports and corresponding config files can be found in the directory 'reports'
