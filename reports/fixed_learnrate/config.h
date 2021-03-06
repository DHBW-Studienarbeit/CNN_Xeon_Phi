#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED


/* [[[cog
import cog
from network_descriptor.NetInstance import net

for key in net._netconfig:
    cog.out('#define ')
    cog.out(key)
    cog.out(' ')
    cog.outl(net._netconfig[key])
]]] */
#define CONFIG_BATCHSIZE 64
#define CONFIG_FLOATTYPE_FLOAT 
#define CONFIG_LEARNING_RATE (0.001f)
#define CONFIG_LEARNRATE_REDUCTION (1.0f) 
//#define CONFIG_REDUCE_CONV_LEARNRATE
#define CONFIG_NUM_OF_ITERATIONS 500
#define CONFIG_TRAININGS_PER_ITERATION 64
#define CONFIG_TESTS_PER_ITERATION 16
#define CONFIG_NUM_DATASETS_PER_FILE 1000
#define CONFIG_DIR_TRAIN "train/"
#define CONFIG_DIR_TEST "test/"
#define CONFIG_NUM_TRAINFILES 55
#define CONFIG_NUM_TESTFILES 10
#define CONFIG_RANDGEN_SEED 777
#define CONFIG_RANDGEN_MEAN 0.0f
#define CONFIG_RANDGEN_RANGESIZE 0.1f
#define CONFIG_MATH_VECT_PARALLEL 
#define CONFIC_VECT_NUM_THREADS 48
#define CONFIG_ARRAY_ALIGNMENT 64
#define CONFIG_NUM_FINAL_TESTS 156
// [[[end]]]

//#define CONFIG_OUTPUT_VERBOSE
//#define CONFIG_QUERY_USER_ACK


#endif /*CONFIG_H_INCLUDED*/
