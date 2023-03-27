#ifndef SAMPLERS_H
#define SAMPLERS_H

#define _USE_MATH_DEFINES 
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iterator>
#include "problems.h"
#include "measurements.h"
#include <mpi.h>


class ISAMPLER{
/*
This is the interface class of the sampling schemes.
The particular samplers are child classes inheriting from this class.
In particular, they need to implement the collect_samples routine.
*/

    private:

        /* Draws a single sampling trajectory. Needs to be defined in child classes. */
        virtual void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas) = 0;     


    public:

        /* Sets up mpi environment and calls "collect_samples" on each process within. Also performs averaging and prints to file. */
        void run_mpi_simulation(int argc, char *argv[], const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const std:: string outputfile, const int t_meas);

        virtual void print_sampler_params();    // Prints sampler hyperparameters, defined in child classes.

        virtual ~ISAMPLER(){};          // Destructor.


};




class OBABO_sampler: public ISAMPLER{
/*
The OBABO splitting scheme.
*/

    private:

        const double T;      // Temperature.
        const double gamma;  // Friction.
        const double h;      // Stepsize.

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas) override; 


    public:

        OBABO_sampler(const double T, const double gamma, const double h): T{T}, gamma{gamma}, h{h} {        // Constructor.
        }; 

        void print_sampler_params() override;  

        ~OBABO_sampler(){};            // Destructor.

};




class BAOAB_sampler: public ISAMPLER{
/*
The BAOAB splitting scheme.
*/

    private:

        const double T;      // Temperature.
        const double gamma;  // Friction.
        const double h;      // Stepsize.

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas) override; 


    public:

        BAOAB_sampler(const double T, const double gamma, const double h): T{T}, gamma{gamma}, h{h} {        // Constructor.
        }; 

        void print_sampler_params() override;  

        ~BAOAB_sampler(){};            // Destructor.

};





class SGHMC_sampler: public ISAMPLER{
/*
The SGHMC sampler (Chen et al. 2014).
Note that whether stochastic gradients are actually used depends on the force routine specified in the problem class passed as an argument.
*/

    private:

        const double T;
        const double gamma;
        const double h;

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas) override;
   
   
    public:

        SGHMC_sampler(double T, double gamma, double h): T{T}, gamma{gamma}, h{h} {        // Constructor.
        }

        void print_sampler_params() override;

        ~SGHMC_sampler(){};              // Destructor

};




class BBK_AMAGOLD_sampler: public ISAMPLER{
/*
The BBK scheme used in the AMAGOLD method (Zhang et al., 2020). Note that it does not use Metropolization.
*/

    private:

        const double T;
        const double gamma;
        const double h;

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas) override;


    public:

        BBK_AMAGOLD_sampler(double T, double gamma, double h): T{T}, gamma{gamma}, h{h} {           // Consructor.
        }

        void print_sampler_params() override;

        ~BBK_AMAGOLD_sampler(){};               // Destructor.

};



#endif // SAMPLERS_H