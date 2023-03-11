#ifndef SAMPLERS_H
#define SAMPLERS_H


#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <math.h> 
#include <iomanip>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iterator>
#include "setup_classes.h"
#include <mpi.h>


class ISAMPLER{

    private:

        virtual measurement collect_samples(const int max_iter, IPROBLEM& POTCLASS, const int randomseed, const int t_meas) = 0;      // actual sampling method. 


    public:

        /* sets up mpi environment and calls "collect_samples" on each process within. Also performs averaging. */
        void run_mpi_simulation(int argc, char *argv[], const int max_iter, IPROBLEM& POTCLASS, const std:: string outputfile, const int t_meas, const bool tavg=0, int n_tavg=10, const int n_dist=1);

        virtual void print_params();    // prints sampler hyperparameters.

        virtual ~ISAMPLER(){};          // destructor.


};




/* The OBABO sampler. It requires the "PROBLEM" class and the "measurement" class in header "setup_classes.h".
   They need to be written by the user and define the sampling problem and what measurements to take. 
   Note that the member functions below require those classes to be structured in a certain way. */

class OBABO_sampler: public ISAMPLER{

    private:

        const double T;
        const double gamma;
        const double h;

        measurement collect_samples(const int max_iter, IPROBLEM& POTCLASS, const int randomseed, const int t_meas) override;  // draws a single sampling trajectory


    public:

        // constructors
        OBABO_sampler(const double T, const double gamma, const double h): T{T}, gamma{gamma}, h{h} {
        }; 

        void print_params() override;

        ~OBABO_sampler(){};    


};




class SGHMC_sampler: public ISAMPLER{

    private:

        const double T;
        const double gamma;
        const double h;

        measurement collect_samples(const int max_iter, IPROBLEM& POTCLASS, const int randomseed, const int t_meas) override;
   
   
    public:

        SGHMC_sampler(double T, double gamma, double h): T{T}, gamma{gamma}, h{h} {
        }

        void print_params() override;

        ~SGHMC_sampler(){};

};




class BBK_AMAGOLD_sampler: public ISAMPLER{

    private:

        const double T;
        const double gamma;
        const double h;

        measurement collect_samples(const int max_iter, IPROBLEM& POTCLASS, const int randomseed, const int t_meas) override;


    public:

        BBK_AMAGOLD_sampler(double T, double gamma, double h): T{T}, gamma{gamma}, h{h} {   
        }

        void print_params() override;

        ~BBK_AMAGOLD_sampler(){};

};








#endif // SAMPLERS_H