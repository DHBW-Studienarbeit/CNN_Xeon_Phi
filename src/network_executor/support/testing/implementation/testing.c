#include <stdio.h>
#include <stdlib.h>


#include "testing.h"


static inline void wait_for_user_ack(void)
{
#ifdef CONFIG_QUERY_USER_ACK
    Int_t ack=0;
    printf("\nAck? ");
    scanf("%d", &ack);
    printf("\n");
    if(ack==0)
    {
        exit(1);
    }
#endif /* CONFIG_QUERY_USER_ACK */
}

void dump_weights(NetState_p netstate)
{
    Int_t i;
    printf("\nweights: ");
    for(i=0; i<NETWORK_WEIGHTS_F_SIZE; i++)
    {
        printf(FLOAT_T_ESCAPE, netstate->weights_f[i]);
        printf(", ");
    }
    printf("\n");
    wait_for_user_ack();
}

/*
void dump_output_labels(NeuronalNetwork_p network, NetState_p netstate, DataSupplier_p supplier)
{
    Int_t i;
    Float_p pos = datasupply_get_output(supplier);
    printf("\noutput, labels: \n");
    for(i=0; i<network->layer_4.output_activation_count; i++)
    {
        printf(FLOAT_T_ESCAPE, netstate->activations[network->layer_4.output_activation_offset + i]);
        printf(", ");
        printf(FLOAT_T_ESCAPE, pos[i]);
        printf("\n");
    }
    printf("\n");
    wait_for_user_ack();
}
*/

void dump_f_array(const char *name, Int_t count, Float_p data)
{
    Int_t i;
    for(i=0;i<count;i++)
    {
        printf("%s[%d] = ", name, i);
        printf(FLOAT_T_ESCAPE, data[i]);
        printf("\n");
    }
    printf("\n");
    wait_for_user_ack();
}

void dump_i_array(const char *name, Int_t count, Int_p data)
{
    Int_t i;
    for(i=0;i<count;i++)
    {
        printf("%s[%d] = %d\n", name, i, data[i]);
    }
    printf("\n");
    wait_for_user_ack();
}
