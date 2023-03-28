#include "samplers.h"
#include "problems.h"
#include "measurements.h"

int main(int argc, char *argv[]){


    double T = 1;           // Sampler hyperparameters.
    double gamma = 20;
    double h = 0.01;

    int iter = 20000;       // Number of iterations (sampler steps).
    int t_meas = 1;          // Take measurement and use it for on-the-fly time-average every t_meas iterations.
    int n_dist = 1;       // Store and print-out any n_dist taken measurement. 
    bool t_avg = false;

    // ### CONSTRUCT ONE OF THE SAMPLERS DEFINED IN "samplers.h". ### //
    OBABO_sampler testsampler(T, gamma, h);    

    
    // ### CONSTRUCT THE PROBLEM TO BE SAMPLED ON, DEFINED IN "problems.h". ### //
    HARMONIC_OSCILLATOR_1D testproblem;  

    
    // ### CONSTRUCT MEASUREMENT OBJECT DEFINED IN "measurements.h". ### //
    MEASUREMENT_DEFAULT RESULTS(n_dist, t_avg);

    std:: string outputfile = "RESULTS.csv";  // Name of output file.


    // ### RUN SAMPLER. ### //
    testsampler.run_mpi_simulation(argc, argv, iter, testproblem, RESULTS, outputfile, t_meas);          


    return 0;


}