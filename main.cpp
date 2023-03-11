#include "samplers.h"
#include "setup_classes.h"

int main(int argc, char *argv[]){

    // we want to sample the double well potential using the OBABO sampler


    double T = 1;           // sampler hyperparameters
    double gamma = 1;
    double h = 0.01;

    int iter = 100;             // no. of iteration
    bool tavg = true;       // perform time-average?
    int n_tavg = 5;         // t-average over the last n_tavg values.
    int t_meas = 1;         // take measurement every t_meas iterations (passed to sampler as well as print functions).
    int n_dist = 1;         // print and t-average (if activated) only every n_dist-th values.

    SGHMC_sampler testsampler(T,gamma,h);

    std:: string filename = "GM_data_5000.csv";
    std:: string outputfile = "RESULTS.csv";
    const int randomseed = 0;
    const int batchsize = 5;

    PROBLEM harmonic_osc;    /* construct object of the problem class defined by the user in header "setup_classes.h". */
    
    testsampler.run_mpi_simulation(argc, argv, iter, harmonic_osc, outputfile, t_meas, tavg, n_tavg, n_dist);  /* run sampler. "measurement" needs to be defined 
                                                                                                            by user in header "setup_classes.h". It holds
                                                                                                            information of what quantities need to be obtained
                                                                                                            by the sampler and how to compute and print them.  */

    return 0;


}