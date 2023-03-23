#include "samplers.h"
#include "setup_classes.h"
#include "measurements.h"

int main(int argc, char *argv[]){

    // we want to sample the double well potential using the OBABO sampler

    double T = 1;           // sampler hyperparameters
    double gamma = 20;
    double h = 0.01;

    int iter = 10000;             // no. of iteration
    int t_meas = 1;         // take measurement every t_meas iterations (passed to sampler as well as print functions).
    int n_dist = 1000;         // print and t-average (if activated) only every n_dist-th values.

    // OBABO_sampler testsampler(T, gamma, h);     // construct OBABO object defined in header "samplers.h"
    // SGHMC_sampler testsampler(T,gamma,h);
    BBK_AMAGOLD_sampler testsampler(T,gamma,h);

    // std:: string filename = "GM_data_5000.csv";
    // const int randomseed = 0;
    // const int batchsize = 5;
    // BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D problem(filename, batchsize);    /* construct object of the problem class defined by the user in header "setup_classes.h". */
    
    HARMONIC_OSCILLATOR_1D problem;
    
    MEASUREMENT_HO_1D RESULTS(2, n_dist);

    std:: string outputfile = "RESULTS.csv";

    testsampler.run_mpi_simulation(argc, argv, iter, problem, RESULTS, outputfile, t_meas);          /* run sampler. "measurement" needs to be defined 
                                                                                                            by user in header "setup_classes.h". It holds
                                                                                                            information of what quantities need to be obtained
                                                                                                            by the sampler and how to compute and print them.  */

    return 0;


}