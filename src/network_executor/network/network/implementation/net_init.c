#include "settings.h"
#include "network.h"



void network_init(void)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net

    i=0
    cog.outl("const Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE] = ")
    cog.outl("{")
    for current in net._layers:
        if current.__class__.__name__ == "MaxPoolingLayer":
            cog.outl("// layer_" + str(i))
            for p_out in range(current.get_output_shape().get_count_probes()):
                for y_out in range(current.get_output_shape().get_count_y()):
                    for x_out in range(current.get_output_shape().get_count_x()):
                        for f_out in range(current.get_output_shape().get_count_features()):
                            for y_filter in range(current._filter_size_y):
                                for x_filter in range(current._filter_size_x):
                                    y_in = current._filter_size_y * y_out + y_filter
                                    x_in = current._filter_size_x * x_out + x_filter
                                    cog.out(str(current.get_input_shape().get_position(p_out, y_in, x_in, f_out)))
                                    cog.out(", ")
                            cog.outl("")
        i=i+1
    cog.outl("};")
    ]]] */

    // [[[end]]]
}
