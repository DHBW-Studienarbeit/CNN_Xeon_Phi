#ifndef DATASUPPLIER_H_INCLUDED
#define DATASUPPLIER_H_INCLUDED

#include "settings.h"


typedef struct {
	Float_p inputs;
	Float_p labels;
	Int_t dataset_index;
	Int_t num_of_files;
} DataSupplier_t, *DataSupplier_p;


#ifdef __cplusplus
extern "C" {
#endif


void datasupply_init(DataSupplier_p supplier, Int_t num_of_files, const char *foldername);
void datasupply_next_batch(DataSupplier_p supplier);
Float_p datasupply_get_input(DataSupplier_p supplier);
Float_p datasupply_get_output(DataSupplier_p supplier);


#ifdef __cplusplus
}
#endif


#endif /* DATASUPPLIER_H_INCLUDED */
