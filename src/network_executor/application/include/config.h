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
#define CONFIG_FLOATTYPE_DOUBLE 
#define CONFIG_LEARNING_RATE (0.01f)
#define CONFIG_LEARNRATE_REDUCTION (0.99f)
#define REDUCE_CONV_LEARNRATE 
#define CONFIG_NUM_OF_ITERATIONS 50
#define CONFIG_TRAININGS_PER_ITERATION 80
#define CONFIG_TESTS_PER_ITERATION 8
#define CONFIG_NUM_DATASETS_PER_FILE 64
#define CONFIG_DIR_TRAIN "train/"
#define CONFIG_DIR_TEST "train/"
#define CONFIG_NUM_TRAINFILES 1
#define CONFIG_NUM_TESTFILES 1
#define CONFIG_RANDGEN_SEED 777
#define CONFIG_RANDGEN_MEAN 0.0f
#define CONFIG_RANDGEN_RANGESIZE 0.1f
#define CONFIG_MATH_VECT_PARALLEL 
#define CONFIC_VECT_NUM_THREADS 60
#define CONFIG_ARRAY_ALIGNMENT 32
// [[[end]]]


//#define CONFIG_QUERY_USER_ACK


#endif /*CONFIG_H_INCLUDED*/
