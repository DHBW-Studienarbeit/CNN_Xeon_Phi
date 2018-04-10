#include <stdio.h>
#include <stdlib.h>

#include "settings.h"
#include "testing.h"
#include "network.h"
#include "datasupplier.h"

extern NeuronalNetwork_t network;
extern DataSupplier_t trainsupplier, testsupplier;


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

void dump_weights(void)
{
    Int_t i;
    printf("\nweights: ");
    for(i=0; i<NETWORK_WEIGHTS_F_SIZE; i++)
    {
        printf(FLOAT_T_ESCAPE, net_weights_f[i]);
        printf(", ");
    }
    printf("\n");
    wait_for_user_ack();
}


void dump_output_labels(void)
{
    Int_t i;
    Float_p pos = datasupply_get_output(&testsupplier);
    printf("\noutput, labels: \n");
    for(i=0; i<network.layer_4.output_activation_count; i++)
    {
        printf(FLOAT_T_ESCAPE, net_activations[network.layer_4.output_activation_offset + i]);
        printf(", ");
        printf(FLOAT_T_ESCAPE, pos[i]);
        printf("\n");
    }
    printf("\n");
    wait_for_user_ack();
}

void dump_array(char *name, Int_t count, Float_p data)
{
    Int_t i;
    for(i=0;i<count;i++)
    {
        printf("%s[%d] = ", name, i);
        printf(FLOAT_T_ESCAPE, data[i]);
        printf("\n");
    }
}
