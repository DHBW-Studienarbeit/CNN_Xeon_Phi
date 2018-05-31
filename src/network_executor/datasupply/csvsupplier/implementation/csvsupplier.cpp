#include "datasupplier.h"
#include "mkl_wrapper.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>


/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("#define NUM_DATASETS_PER_BATCH " + str(net.get_input_shape().get_count_probes()))
cog.outl("#define DATASET_INPUT_SIZE " + str(net.get_input_shape().get_count_x() \
 											* net.get_input_shape().get_count_y() \
											* net.get_input_shape().get_count_features()))
cog.outl("#define DATASET_OUTPUT_SIZE " + str(net.get_output_shape().get_count_x() \
 											* net.get_output_shape().get_count_y() \
											* net.get_output_shape().get_count_features()))
]]] */
#define NUM_DATASETS_PER_BATCH 100
#define DATASET_INPUT_SIZE 784
#define DATASET_OUTPUT_SIZE 10
// [[[end]]]


static inline void datasupply_load_file(DataSupplier_p supplier, const char *foldername, Int_t file_index)
{
    Int_t i,j;
    std::string csv_file = foldername + std::to_string(file_index) + ".csv";
	std::ifstream infile(csv_file);
	for(j=0; j<CONFIG_NUM_DATASETS_PER_FILE; j++)
	{
		std::string line, cell;
		std::getline(infile,line);
        std::stringstream lineStream(line);
    	for(i=0; i<DATASET_INPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
#ifdef CONFIG_FLOATTYPE_DOUBLE
            supplier->inputs[
             file_index * CONFIG_NUM_DATASETS_PER_FILE * DATASET_INPUT_SIZE
             + j * DATASET_INPUT_SIZE + i] = std::stod(cell);
#endif
#ifdef CONFIG_FLOATTYPE_FLOAT
            supplier->inputs[
             file_index * CONFIG_NUM_DATASETS_PER_FILE * DATASET_INPUT_SIZE
             + j * DATASET_INPUT_SIZE + i] = std::stof(cell);
#endif
    	}
    	for(i=0; i<DATASET_OUTPUT_SIZE; i++)
    	{
    		std::getline(lineStream,cell, ',');
#ifdef CONFIG_FLOATTYPE_DOUBLE
            supplier->labels[
             file_index * CONFIG_NUM_DATASETS_PER_FILE * DATASET_OUTPUT_SIZE
             + j * DATASET_OUTPUT_SIZE + i] = std::stod(cell);
#endif
#ifdef CONFIG_FLOATTYPE_FLOAT
            supplier->labels[
             file_index * CONFIG_NUM_DATASETS_PER_FILE * DATASET_OUTPUT_SIZE
             + j * DATASET_OUTPUT_SIZE + i] = std::stof(cell);
#endif
    	}
	}
}


void datasupply_init(DataSupplier_p supplier, Int_t num_of_files, const char *foldername)
{
    Int_t file_index;
    supplier->inputs = MATH_MALLOC_F_ARRAY(num_of_files * CONFIG_NUM_DATASETS_PER_FILE * DATASET_INPUT_SIZE);
    supplier->labels = MATH_MALLOC_F_ARRAY(num_of_files * CONFIG_NUM_DATASETS_PER_FILE * DATASET_OUTPUT_SIZE);
    for(file_index=0; file_index<num_of_files; file_index++)
    {
        datasupply_load_file(supplier, foldername, file_index);
    }
    supplier->dataset_index = 0;
	supplier->num_of_files = num_of_files;
}

void datasupply_next_batch(DataSupplier_p supplier)
{
    supplier->dataset_index += NUM_DATASETS_PER_BATCH;
    if(supplier->dataset_index >= supplier->num_of_files * CONFIG_NUM_DATASETS_PER_FILE - NUM_DATASETS_PER_BATCH)
    {
        supplier->dataset_index = 0;
    }
}

Float_p datasupply_get_input(DataSupplier_p supplier)
{
    return supplier->inputs + DATASET_INPUT_SIZE * supplier->dataset_index;
}

Float_p datasupply_get_output(DataSupplier_p supplier)
{
    return supplier->labels + DATASET_OUTPUT_SIZE * supplier->dataset_index;
}
