#include "datasupplier.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


void datasupply_load_batch(DataSupplier_p supplier)
{
    std::string csv_file = supplier->foldername + "/"
                         + std::to_string(this->file_index) + ".csv";
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
