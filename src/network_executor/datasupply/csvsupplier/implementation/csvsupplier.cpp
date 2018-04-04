#include "datasupplier.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


static inline void datasupply_load_batch(DataSupplier_p supplier)
{
    std::string csv_file = supplier->foldername + "/"
                         + std::to_string(supplier->file_index) + ".csv";
	std::ifstream infile(csv_file);
	for(int i=0; i<NUM_DATASETS_PER_FILE; i++)
	{
		std::string line, cell;
		std::getline(infile,line);
        std::stringstream          lineStream(*line);
    	for(int i=0; i<INPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
    		supplier->inputs[i] = std::stof(cell);
    	}
    	for(int i=0; i<OUTPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
    		supplier->labels[i] = (float)std::stod(cell);
    	}
	}
    supplier->file_index++;
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
        datasupply_load_batch(supplier);
    }
}

Float_p datasupply_get_input(DataSupplier_p supplier)
{
    return supplier->inputs[DATASET_INPUT_SIZE * NUM_DATASETS_PER_BATCH
                            * (supplier->batch_index) ];
}

Float_p datasupply_get_output(DataSupplier_p supplier)
{
    return supplier->inputs[DATASET_OUTPUT_SIZE * NUM_DATASETS_PER_BATCH
                            * (supplier->batch_index) ];
}
