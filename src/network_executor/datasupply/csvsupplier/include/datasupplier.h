#ifndef DATASUPPLIER_H_INCLUDED
#define DATASUPPLIER_H_INCLUDED

#include "settings.h"

#define NUM_DATASETS_PER_FILE   1000
#define FOLDERNAME_MAX_LENGTH   100

#define NUM_DATASETS_PER_BATCH	20
#define NUM_BATCHES_PER_FILE	50
#define DATASET_INPUT_SIZE		786
#define DATASET_OUTPUT_SIZE		10


typedef struct
{
	Float_t inputs[NUM_DATASETS_PER_FILE*DATASET_INPUT_SIZE];
	Float_t labels[NUM_DATASETS_PER_FILE*DATASET_OUTPUT_SIZE];
	Int_t batch_index;
	Int_t file_index;
	Int_t num_of_files;
	char foldername[FOLDERNAME_MAX_LENGTH];
} DataSupplier_t, *DataSupplier_p;


#ifdef __cplusplus
extern "C" {
#endif

void datasupply_next_batch(DataSupplier_p supplier);
Float_p datasupply_get_input(DataSupplier_p supplier);
Float_p datasupply_get_output(DataSupplier_p supplier);

#ifdef __cplusplus
}
#endif


#endif /* DATASUPPLIER_H_INCLUDED */
