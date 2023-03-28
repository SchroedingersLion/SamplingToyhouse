#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>


class IMEASUREMENT{
/*
This is the interface class of the measurements.
Particular measurement classes are child classes inheriting from this class.
In particular, they need to implement the compute_sample routine.
*/

    public:

        // Constructor.
        /* 
        no_observables gives the number of observables the measuremement object is going to collect (eg. = 2 if one only wants to store kinetic and configurational temperature). 
        n_dist gives the frequency by which measurements are printed out to file (eg. if n_dist = 5, any 5th taken sample will be printed).
        */
        IMEASUREMENT(const size_t no_observables, const size_t n_dist, const bool time_average = true)                         
            : measured_values{std:: vector < std::vector <float> > {no_observables}}, measured_values_AVG{std:: vector < std::vector <float> > {no_observables}}, 
              summed_values{std:: vector <double> (no_observables,0)}, samples{std:: vector <double> (no_observables,0)}, ctr{0}, n_dist{n_dist}, time_average{time_average} {};  



        /* Needs to be overridden by child classes. Computes observables from given positions, velocities, and forces, and stores them in samples vector (see examples below). */
        virtual void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) = 0;


        /*
        Method called by the samplers to take a measurement. It calls the user specified compute_sample routine to obtain oversables, performs the time-average, 
        and stores any n_dist results in the measured_values vector for future print to file. 
        */
        void take_measurement(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces){

            if ( time_average == true ){

                compute_sample(parameters, velocities, forces);             // Compute samples...

                for (size_t i = 0;  i < summed_values.size();  ++i){        // ...add them to sum for t-average...
                    summed_values[i] += samples[i];
                }

                if ( ctr % n_dist == 0 ){                                   // ...but only perform average and store them for print-out 
                                                                            //     any n_dist taken measurements.
                    
                    for (size_t i = 0;  i < summed_values.size();  ++i){
                        measured_values[i].push_back(summed_values[i] / (ctr+1));
                    }
                
                }

            }

            else {                                                              // No t-average...

                if ( ctr % n_dist == 0 ){

                    compute_sample(parameters, velocities, forces);             // ...so only compute samples when they are
                                                                                // going to be printed out later.
                    
                    for (size_t i = 0;  i < samples.size();  ++i){
                        measured_values[i].push_back(samples[i]);
                    }
                    
                }

            }

            ++ctr;

        };


        /* 
        Method to print out the time-and process-averaged results. 
        Called by the samplers at the end of the simulation.
        */
        void print_to_csv(const int t_meas, const std:: string outputname){    

            std:: ofstream file{outputname};
            std:: cout << "Writing to file...\n";

            for ( size_t i = 0; i<measured_values_AVG[0].size(); ++i )
            {
                file << i*t_meas*n_dist << " ";
                for ( size_t j = 0; j<measured_values_AVG.size(); ++j ){
                    file << measured_values_AVG[j][i] << " ";  
                }
                file << "\n";
            }

            file.close();

        }


        virtual ~IMEASUREMENT(){};                      // Destructor.


        
        std:: vector < std::vector <float> > measured_values;       // 2D vector, storing the measured (and t-averaged) observables in its rows.
        std:: vector < std::vector <float> > measured_values_AVG;   // 2D vector, storing the t- and process-averaged observables in its rows.
        int ctr{0};                                                 // Help variable to count the number of measurements taken.
        const size_t n_dist;                                        // Any n_dist taken measurements will be stored and printed out by print-out routine.
        std:: vector <double> summed_values;                        // Help vector to add taken measurements for time-average.
        std:: vector <double> samples;                              // Vector storing the observables computed from a single set of model parameters at a given time (positions, velocities, forces).
        bool time_average;                                          // Decides whether measurements are time-averaged.

};



class MEASUREMENT_DEFAULT: public IMEASUREMENT{
/* 
This is a default measurement class. It can be used with any model/problem. 
It stores 3 observables: The first model parameter (i.e. the first position coordinate), the kinetic, and the configurational temperature given by Force \cdot Position.
*/

    public:
        
        // Constructor. 
        /* 
        Note that it calls the interface class' constructor and passes 3 as input for "no_observables", determining the size of the vector "samples".
        This is somewhat unsafe, as the number passed has to match the number of entries "samples" is expected to have in the "compute_sample" routine below.
        It needs to be changed in the future.
        */ 

        MEASUREMENT_DEFAULT(const size_t n_dist, const bool t_avg = true)
            : IMEASUREMENT(3, n_dist, t_avg) {};
        
        
        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  

            samples[0] = parameters[0];
            samples[1] = 1./parameters.size() * (velocities[0]*velocities[0] + velocities[1]*velocities[1]);
            samples[2] = - 1./parameters.size() * (parameters[0]*forces[0] + parameters[1]*forces[1]); 
        
        };

};



class MEASUREMENT_HO_1D: public IMEASUREMENT{
/*
Measurement class used for the 1D harmonic oscillator. 
It collects the same observables as in Leimkuhler & Matthews 2013, i.e. q^2, p^2, and qp. 
*/

    public:

        MEASUREMENT_HO_1D(const size_t n_dist, const bool t_avg = true)
            : IMEASUREMENT(3, n_dist, t_avg) {};

        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  
                                                                                                                   
            samples[0] = parameters[0]*parameters[0];
            samples[1] = velocities[0]*velocities[0];
            samples[2] = parameters[0]*velocities[0];
        
        };

};



class MEASUREMENT_2D: public IMEASUREMENT{
/*
Measurement class useful for 2-dimensional problems. It collects the two coordinates, kinetic and config. temperature.
*/

    public:

        MEASUREMENT_2D(const size_t n_dist, const bool t_avg = true)
            : IMEASUREMENT(4, n_dist, t_avg) {};

        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  
                                                                                                                   
            samples[0] = parameters[0];
            samples[1] = parameters[1];
            samples[2] = 0.5 * (velocities[0]*velocities[0] + velocities[1]*velocities[1]);
            samples[3] = -0.5 * (parameters[0]*forces[0] + parameters[1]*forces[1]); 
        
        };

};


#endif // MEASUREMENTS_H