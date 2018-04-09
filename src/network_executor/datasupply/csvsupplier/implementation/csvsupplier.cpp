#include "datasupplier.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>


static inline void datasupply_load_file(DataSupplier_p supplier)
{
    Int_t i,j;
    std::string csv_file = supplier->foldername
                         + std::to_string(supplier->file_index) + ".csv";
	std::ifstream infile(csv_file);
	for(j=0; j<NUM_DATASETS_PER_FILE; j++)
	{
		std::string line, cell;
		std::getline(infile,line);
        std::stringstream lineStream(line);
    	for(i=0; i<DATASET_INPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
#ifdef CONFIG_FLOATTYPE_DOUBLE
            supplier->inputs[j * DATASET_INPUT_SIZE + i] = std::stod(cell);
#endif
#ifdef CONFIG_FLOATTYPE_FLOAT
            supplier->inputs[j * DATASET_INPUT_SIZE + i] = std::stof(cell);
#endif
    	}
    	for(i=0; i<DATASET_OUTPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
#ifdef CONFIG_FLOATTYPE_DOUBLE
            supplier->labels[j * DATASET_OUTPUT_SIZE + i] = std::stod(cell);
#endif
#ifdef CONFIG_FLOATTYPE_FLOAT
            supplier->labels[j * DATASET_OUTPUT_SIZE + i] = std::stof(cell);
#endif
    	}
	}
}


void datasupply_init(DataSupplier_p supplier, Int_t num_of_files, char *foldername)
{
    supplier->batch_index = -1;
	supplier->file_index = 0;
	supplier->num_of_files = num_of_files;
	std::strcpy(supplier->foldername, foldername);
    datasupply_load_file(supplier);
}

void datasupply_next_batch(DataSupplier_p supplier)
{
    supplier->batch_index++;
    if(supplier->batch_index >= NUM_BATCHES_PER_FILE)
    {
        supplier->batch_index = 0;
        supplier->file_index++;
        if(supplier->file_index >= supplier->num_of_files)
        {
            supplier->file_index = 0;
        }
        datasupply_load_file(supplier);
    }
}

Float_p datasupply_get_input(DataSupplier_p supplier)
{
    return supplier->inputs + DATASET_INPUT_SIZE * NUM_DATASETS_PER_BATCH
                            * (supplier->batch_index);
}

Float_p datasupply_get_output(DataSupplier_p supplier)
{
    return supplier->labels + DATASET_OUTPUT_SIZE * NUM_DATASETS_PER_BATCH
                            * (supplier->batch_index);
}
