#include <stdio.h>

#include "settings.h"
#include "trainsession.h"
#include "testsession.h"
#include "network.h"
#include "weightgenerator.h"
#include "testing.h"
#include "mathematics.h"



DataSupplier_t trainsupplier, testsupplier;
NeuronalNetwork_t network;


int main(void)
{
    TestResult_t testresult;
    Int_t iteration;
    Float_t last_cost;
    Float_t learning_rate = CONFIG_LEARNING_RATE / CONFIG_BATCHSIZE;
#ifdef CONFIG_OUTPUT_VERBOSE
    printf("do some initialization stuff\n");
#endif
    // allocate memory for network execution
    network_init(&network);
    // generate weights for the network
    weightgen_generate(NETWORK_WEIGHTS_F_SIZE, network.weights_f);
    // initialize suppliers to read input data and labels
    datasupply_init(&trainsupplier, CONFIG_NUM_TRAINFILES, CONFIG_DIR_TRAIN);
    datasupply_init(&testsupplier, CONFIG_NUM_TESTFILES, CONFIG_DIR_TEST);
    // test with random weights first; just for later comparison
    exec_testsession(&network, &testsupplier, CONFIG_TESTS_PER_ITERATION, &testresult);
#ifdef CONFIG_OUTPUT_VERBOSE
    printf("Iteration 0: ");
    printf(FLOAT_T_ESCAPE, testresult.accuracy);
    printf("\t");
    printf(FLOAT_T_ESCAPE, testresult.cost);
    printf("\n");
#endif
    last_cost = testresult.cost;
    // do iterations consisting of training and testing
    for(iteration=1; iteration<=CONFIG_NUM_OF_ITERATIONS; iteration++)
    {
        exec_trainsession(&network, &trainsupplier, CONFIG_TRAININGS_PER_ITERATION, learning_rate);
        exec_testsession(&network, &testsupplier, CONFIG_TESTS_PER_ITERATION, &testresult);
#ifdef CONFIG_OUTPUT_VERBOSE
        printf("Iteration %d: ", iteration);
        printf(FLOAT_T_ESCAPE, testresult.accuracy);
        printf("\t");
        printf(FLOAT_T_ESCAPE, testresult.cost);
        printf("\n");
#endif
#ifdef CONFIG_LEARNRATE_REDUCTION
        if(testresult.cost > last_cost)
        {
            learning_rate *= CONFIG_LEARNRATE_REDUCTION;
#ifdef CONFIG_OUTPUT_VERBOSE
            printf("Setting learning rate to ");
            printf(FLOAT_T_ESCAPE, learning_rate);
            printf("\n");
#endif
        }
        last_cost = testresult.cost;
#endif
    }

    exec_testsession(&network, &testsupplier, CONFIG_NUM_FINAL_TESTS, &testresult);
#ifdef CONFIG_OUTPUT_VERBOSE
    printf("\n+++ SUMMARY +++\n");
#endif
    printf("%d Iterations performed\nfinal accuracy: ", CONFIG_NUM_OF_ITERATIONS);
    printf(FLOAT_T_ESCAPE, testresult.accuracy);
    printf("\nfinal cost per picture: ");
    printf(FLOAT_T_ESCAPE, testresult.cost);
    printf("\nfinal learning rate: ");
    printf(FLOAT_T_ESCAPE, learning_rate);
    printf("\n");

    return 0;
}
