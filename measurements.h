#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>


class IMEASUREMENT{

    public:

        // constructor. only initializes members. 
        IMEASUREMENT(const size_t no_params, const size_t n_dist)                         
            : measured_values{std:: vector < std::vector <float> > {no_params}}, measured_values_AVG{std:: vector < std::vector <float> > {no_params}}, 
              summed_values{std:: vector <double> (no_params,0)}, samples{std:: vector <double> (no_params,0)}, ctr{0}, n_dist{n_dist} {};  


        virtual void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) = 0;


        void take_measurement(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces){

            compute_sample(parameters, velocities, forces);

            for (size_t i = 0;  i < summed_values.size();  ++i){
                summed_values[i] += samples[i];
            }

            if ( ctr % n_dist == 0 ){
                
                for (size_t i = 0;  i < summed_values.size();  ++i){
                    measured_values[i].push_back(summed_values[i] / (ctr+1));
                }
                
            }

            ++ctr;

        };


        void print_to_csv(const int t_meas, const std:: string outputname){    

            std:: ofstream file{outputname};
            std:: cout << "Writing to file...\n";

            // print out every entry 
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


        virtual ~IMEASUREMENT(){};                      // destructor.


        
        std:: vector < std::vector <float> > measured_values;   // vector of vectors, storing the measured observables in its rows.
        std:: vector < std::vector <float> > measured_values_AVG;
        int ctr{0};
        const size_t n_dist;
        std:: vector <double> summed_values;
        std:: vector <double> samples;

};


// stadard measurement class
class MEASUREMENT_DEFAULT: public IMEASUREMENT{

    public:
        
        MEASUREMENT_DEFAULT(const size_t n_dist)
            : IMEASUREMENT(3, n_dist) {};
        
        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  

            samples[0] = parameters[0];
            samples[1] = 0.5 * (velocities[0]*velocities[0] + velocities[1]*velocities[1]);
            samples[2] = -0.5 * (parameters[0]*forces[0] + parameters[1]*forces[1]); 
        
        };

};


// measurement class for the harmonic oscillator.
class MEASUREMENT_HO_1D: public IMEASUREMENT{

    public:

        MEASUREMENT_HO_1D(const size_t n_dist)
            : IMEASUREMENT(3, n_dist) {};

        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  
                                                                                                                   
            samples[0] = parameters[0]*parameters[0];
            samples[1] = velocities[0]*velocities[0];
            samples[2] = parameters[0]*velocities[0];
        
        };

};


// measurement class for the curved channel 2D double well 
class MEASUREMENT_DW_2D_CURVED: public IMEASUREMENT{

    public:

        MEASUREMENT_DW_2D_CURVED(const size_t n_dist)
            : IMEASUREMENT(4, n_dist) {};

        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, const std:: vector <double>& forces) override {  
                                                                                                                   
            samples[0] = parameters[0];
            samples[1] = parameters[1];
            samples[2] = 0.5 * (velocities[0]*velocities[0] + velocities[1]*velocities[1]);
            samples[3] = -0.5 * (parameters[0]*forces[0] + parameters[1]*forces[1]); 
        
        };

};


#endif // MEASUREMENTS_H