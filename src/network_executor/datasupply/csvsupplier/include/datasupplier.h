#ifndef DATASUPPLIER_H_INCLUDED
#define DATASUPPLIER_H_INCLUDED

#include "settings.h"

#define NUM_PICS_PER_FILE       1000
#define FOLDERNAME_MAX_LENGTH   100

typedef struct
{
	Float_t images[PICS_PER_FILE];
	int next_index;
	int file_index;
	int num_of_files;
	char foldername[];
} DataSupplier_t, *DataSupplier_p;

#endif /* DATASUPPLIER_H_INCLUDED */
