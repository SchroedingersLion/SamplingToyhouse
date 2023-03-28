# SamplingToyhouse
An efficient framework to test novel sampling schemes on simple problems.

## What is it for?
We are interested in drawing samples from the continuous density $$\rho(q)\propto e^{-U(q)}$$ 
with given potential energy function $U(q)$. Our sampling methods of choice are Markov Chain Monte Carlo (MCMC) methods built on either 
Langevin or Hamiltonian dynamics. 
Since there is an abundance of different MCMC sampling schemes out there, and researchers keep developing novel algorithms with each passing day, 
it would be useful to have a library collecting these schemes in a unified way. Moreover, as novel algorithms need to be tested 
on various problems specified by different $U(q)$, the library of sampling schemes should be accompanied by a library of test problems to run them on. 
This would be most useful to mathematical researchers (professors and students alike!) working on novel MCMC methods. 

While there are many libraries and frameworks out there that provide state-of-the art sampling schemes, they often suffer from one or more of 
the following drawbacks:

1. They are too big and complex so that it is difficult for the researcher to know what exactly is going on under the hood. 

2. They are bloated with tools and utilities that are not of interest to the researcher. 

3. They only offer schemes that had been widely tested and are already in wide use (as opposed to the most recent, most novel schemes that sampling researchers work on). 

4. They don't offer the option to specify arbitrary test problems $U(q)$ or observables to be collected. 

5. It takes too much programming / software engineering experience to contribute to their development (eg. by adding new samplers).


This project tries to be the answer to these points. It targets the mathematical, statistical, or physical science researcher that works on 
novel sampling schemes that need to be tested on simple problems. It is a minimalistic framework and aims to be both efficient and easy to use. 
Moreover, it is straight-forwardly expandable. Specifically, it has the following features:

• A library of efficiently implemented MCMC sampling algorithms that is easily expandable.

• A library of various test problems governed by different choices of $U(q)$ that is easily expandable.

• The straight-forward specification of observables that are to be collected by the samplers.

• The straight-forward running of various samplers collecting various observables on various problems.

• Support for parallelization using the message-passing-interface (MPI) library.

**A typical usecase would be**: A researcher develops a new sampling algorithm to sample $\rho$. He wants to examine its properties on simple test problems. 
The advantage of simple problems is that **a)** the corresponding function $U(q)$ is often mathematically benign to the point where the system 
can be more readily treated by theory, and **b)** the computational cost to obtain results is low. Examples of problems like this are the harmonic 
oscillator in small dimensions, 2-dimensional double well surfaces, or even data science problems such as simple Bayesian logistic regressions.  
Once the researcher has an idea of how his new scheme behaves in these settings, he wants to compare its performance to other samplers, some of which are 
already in world-wide use, others are just as novel and recent as the scheme he just developed.  
He wants to quickly pick and run various samplers on 
various simple problems to compare their performance. Moreover, he wants to be able to freely pick which observables to collect on each problem.  
The results of these experiments will guide his understanding of how different samplers work, how and why they differ, and which ones are the most efficient.

## What is it not for?
The simplicity by which new sampling problems $U(q)$ can be defined in this framework comes at the price of the problems having to be comparably simple. 
In particular, the gradients of the functions $U(q)$ need to be hardcoded into the problems. Problems that are built on complex models such as deep neural 
networks will be very tedious to implement here, as this framework does not offer automatic backpropagation nor any of the fancy regularization methods 
often used in deep learning. Another example of problems that are “too big” are larger molecular dynamics (MD) simulations. While the hardcoding 
of the gradients is typically not an issue there, efficiency is. When simulating a system of $N$ interacting particles, one often needs to employ 
high-performance-computing techniques beyond what this framework has to offer. Therefore, if one is interested in large-scale sampling problems, 
one should resort to using other libraries/frameworks or write their own code from scratch. This means that this framework is most likely useless to 
the machine learning researcher working with big-data models, the molecular researcher working with large-scale MD simulations, and certainly to 
companies trying to solve industrial problems. It is the Sampling-TOY-house, after all.

## The Three Building Blocks
Here we give a high-level overview of the three building blocks the framework is comprised of: The problem library, the measurement library, 
and the sampler library. Each library holds an interface class that serves as a base class that all other classes in the library inherit from. 
Users will generally not need to modify the interface classes. However, they will need to understand them to a certain degree if they want to add 
their own elements to the system, eg. if they want to add new sampling schemes.

### The Problem Library
The problem library is given by the header file **problems.h**. It holds the sampling problems, each of which is defined as an individual class inheriting 
from an abstract parent class named `IPROBLEM`. The latter's definition can be found in the file: 
```C++
class IPROBLEM {
/* 
The interface class of the problems to sample on (eg. double wells, harmonic oscillators, ...).
The particular problems are child classes inheriting from this class.
In particular, they need to implement the compute_force() routine.
*/

    public:

       std:: vector <double> parameters;            // Parameters, velocities, forces. 
       std:: vector <double> velocities;       
       std:: vector <double> forces;           
        
       // Constructor. Initializes the trajectories. 
       IPROBLEM(const std:: vector <double>& init_params, const std:: vector <double>& init_velocities)                         
           : parameters{init_params}, velocities{init_velocities}, forces{std:: vector <double> (parameters.size(),0)} {};   

       // Fills the force vector in each sampler iteration. Needs to be defined by the child classes.
       virtual void compute_force() = 0;           

       virtual ~IPROBLEM(){};                      // Destructor.

};
```
It holds three member vectors, `parameters` (i.e. the position q), `velocities`, and `forces`. Apart from constructor and destructor, 
it only holds one method, the `compute_force` routine which is a purely virtual function and needs to be defined in every child class. 
This function will fill the force vector given the parameters, and it is called by the sampling schemes. We will see how to create a particular sampling 
problem from this interface class in the next section.

### The Measurement Library
The measurement library is given by the header file **measurements.h**. The classes in this file specify different measurement processes, 
in particular which observables to collect on a given sampling problem.  
All measurement classes are derived from an abstract parent class `IMEASUREMENT`. The latter's definition can be found in the file:
```C++
class IMEASUREMENT{
/*
This is the interface class of the measurements.
Particular measurement classes are child classes inheriting from this class.
In particular, they need to implement the compute_sample routine.
*/

    public:
        
        /*
	Constructor.
        no_observables gives the number of observables the measuremement object is going to collect 
	(eg. = 2 if one only wants to store kinetic and configurational temperature). 
        n_dist gives the frequency by which measurements are printed out to file 
	(eg. if n_dist = 5, any 5th taken sample will be printed).
        */
        IMEASUREMENT(const size_t no_observables, const size_t n_dist, const bool time_average = true)                         
            : measured_values{std:: vector < std::vector <float> > {no_observables}}, 
	      measured_values_AVG{std:: vector < std::vector <float> > {no_observables}}, 
              summed_values{std:: vector <double> (no_observables,0)}, 
	      samples{std:: vector <double> (no_observables,0)}, ctr{0}, n_dist{n_dist}, 
	      time_average{time_average} 
	      {};  


        /* 
	Needs to be overridden by child classes. Computes observables from given positions, velocities, and forces, 
	and stores them in samples vector (see examples below). 
	*/
        virtual void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, 
				    const std:: vector <double>& forces) = 0;


        /*
        Method called by the samplers to take a measurement. It calls the user specified compute_sample routine 
	to obtain obersables, performs the time-average, and stores any n_dist results in the 
	measured_values vector for future print to file. 
        */
        void take_measurement(const std:: vector <double>& parameters, const std:: vector <double>& velocities, 
			     const std:: vector <double>& forces){

            if ( time_average == true ){

                compute_sample(parameters, velocities, forces);             // Compute samples...

                for (size_t i = 0;  i < summed_values.size();  ++i){        // ...add them to sum for t-average...
                    summed_values[i] += samples[i];
                }

                if ( ctr % n_dist == 0 ){                                   // ...but only perform average and store them for print-out 
                                                                            //    any n_dist taken measurements.
                    
                    for (size_t i = 0;  i < summed_values.size();  ++i){
                        measured_values[i].push_back(summed_values[i] / (ctr+1));
                    }
                
                }

            }

            else {                                                         // No t-average...

                if ( ctr % n_dist == 0 ){

                    compute_sample(parameters, velocities, forces);        // ...so only compute samples when 
		    							   // they are  going to be printed out later.
                    
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


        // 2D vector, storing the measured (and t-averaged) observables in its rows.
        std:: vector < std::vector <float> > measured_values;       
        
	// 2D vector, storing the t- and process-averaged observables in its rows.
	std:: vector < std::vector <float> > measured_values_AVG;  
        
	// Help variable to count the number of measurements taken.
	int ctr{0};                                                 
        
	// Any n_dist taken measurements will be stored and printed out by print-out routine.
	const size_t n_dist;                                        
        
	// Help vector to add taken measurements for time-average.
	std:: vector <double> summed_values;                        
        
	/* Stores the observables computed from a single set of model parameters 
	at a given time (positions, velocities, forces). */
	std:: vector <double> samples;                              
        
	bool time_average;    // Decides whether measurements are time-averaged.

};
```

Note that the implementation of the member functions is not sourced out to an additional **.cpp** file (as is often considered 'best practice') 
in order to allow the users to define their own measurement classes by modifying just a single file.  
We see that the parent measurement class holds several member variables that all are inherited by the particular measurement child classes. In particular, 
they are used to obtain and store the (averaged) observables during sampling.  
The method `compute_sample` is purely virtual and needs to be implemented by all measurement child classes (see below for the default measurement class).
Given a set of parameters, velocities, and forces, that is passed as input arguments, it computes a set of observables and stores them in the `samples`
member vector.  
The class comes with two predefined methods, `print_to_csv` (which prints the averaged observables collected during sampling to a **.csv** file) 
and `take_measurement`. The latter routine is called by the samplers any `t_meas` steps (see the next section). It computes a sample of observables
via `compute_sample` and performs on-the-fly time-averaging if the Boolean member variable `time_average` is set to `true`. However, 
it only stores every `n_dist` taken observable set to the result vector `measured_values`. The entries of that vector are going to be 
process-averaged and printed to a file by the samplers at the end of the simulation. In other words, if one wanted to have an output file holding 
trajectory data for every 1000th sampler step, one needs to pick `t_meas` and `n_dist` such that 1000= `t_meas` x `n_dist`.  

The header **measurements.h** already holds a predefined measurement class, named `MEASUREMENT_DEFAULT`, that is usable with any sampling problem. Its implementation of `compute_sample` collects the first position coordinate, the kinetic temperature  
$T_{kin}=\frac{1}{N_{d}}p\cdot p$, and the configurational temperature $T_{conf}=-\frac{1}{N_{d}}q\cdot F$ as samples, where $N_d$ is the number of degrees of freedom.  
Its implementation is displayed in the next section as an example to create measurement classes derived from `IMEASUREMENT`.

### The Sampler Library
The sampler library holds the various sampling schemes. It consists of two files: **samplers.h**, which gives the class definitions of the samplers, 
and **samplers.cpp**, which holds the implementation of the member functions. All samplers are classes directly derived from an interface parent class 
called `ISAMPLER`. Its definition is found in the header file: 
```C++
class ISAMPLER{
/*
This is the interface class of the sampling schemes.
The particular samplers are child classes inheriting from this class.
In particular, they need to implement the collect_samples routine.
*/

    private:

        /* Draws a single sampling trajectory. Needs to be defined in child classes. */
        virtual void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, 
				     const int randomseed, const int t_meas) = 0;     


    public:

        /* 
	Sets up mpi environment and calls "collect_samples" on each process within. 
	Also performs averaging and prints to file. 
	*/
        void run_mpi_simulation(int argc, char *argv[], const int max_iter, IPROBLEM& problem, 
				IMEASUREMENT& RESULTS, const std:: string outputfile, const int t_meas);

        virtual void print_sampler_params();    // Prints sampler hyperparameters, defined in child classes.

        virtual ~ISAMPLER(){};          // Destructor.


};

```
The function `draw_trajectory` is purely virtual and thus needs to be implemented in all child classes. It draws a single sampling trajectory on 
the sampling problem passed as an argument, and collects the observables as specified in the measurement object passed as an argument.  
The function _**run_mpi_simulation**_ sets up the MPI environment, calls `draw_trajectory` on every process, averages the results, and prints them to 
a **.csv file**. It is inherited by all sampler child classes as is and **must not** be modified. 

## How to use it
In order to use the SamplingToyhouse, we need to download the following files from the Github repository: **main.cpp**, **samplers.cpp**, **samplers.h**, 
**measurements.h**, **problems.h** to a common folder.  
Rather than manually downloading, we could also clone the whole respository, which will be the method of choice if we want to contribute to the project 
in the long run.  

After adjusting the main file (which includes the two header files), the code needs to be compiled via the MPI compiler wrapper provided by the 
respective MPI library being used on the system. On a Linux machine using the _OpenMPI_ library and the _GCC_ compiler, the compilation is invoked via 
`mpicxx -O3 -o main.exe main.cpp samplers.cpp`. This will create an executable called **main.exe** which can be run via `mpirun -n N main.exe`, where N needs to be replaced by the number of MPI processes that are supposed to be launched. Each process will draw a single trajectory.

### With predefined samplers, problems, and measurement objects
In the simplest case, a user might want to sample one of the problems that are already implemented in the problem library, using a sampler that 
is already implemented in the sampler library. If he is also happy with the observables taken by one of the measurement classes that are already 
implemented, all he has to do is to write up a main file which loads and initializes the corresponding objects from the libraries, and call the 
`run_mpi_simulation` routine.  
An example main file for sampling from the 1-dimensional harmonic oscillator using the OBABO splitting scheme could look as follows: 
```C++
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
```
In this situation, the user was happy to use the default measurement class, which uses the first (and in the 1D-case only) position coordinate as well 
as the kinetic and configurational temperature as observables, and saves the results to a file named **RESULTS.csv**. The Boolean `time_average` is set
to `false`, meaning that no time-average is taken. Running the code via `mpirun -n 1 main.exe` draws a single sampling trajectory. Using a simple 
python script, we plot the results:  

![HO_demo_single](https://user-images.githubusercontent.com/70909827/228239263-b2089ae9-3b68-42ea-a2b1-700c742ae102.png)
We see wild oscillations (pun intended) in the observables coming from the fact that we did not use any kind of averaging.  
Rerunning the code with 100 processes will lead to an average over 100 different trajectories, generating the following results:  

![HO_demo_100traj](https://user-images.githubusercontent.com/70909827/228239583-c03f90a4-f8a1-48b1-9db4-7278f224b58a.png)
The observables start to converge to the expected values (0 for the x-coordinate, 1 for the two temperatures). If we now switch on the time average by 
setting `time_average` to `true` (don't forget to recompile!) we can improve the results even further:  

![HO_demo_100traj_time](https://user-images.githubusercontent.com/70909827/228239644-0d441d5b-9015-41bc-b0d2-76bd2be6a4d9.png)


At this point, the input parameters passed to the constructors of the samplers, the measurement objects, and the problems will need to be inferred by 
the user from looking at their respective definitions in the header files (later, a documentation for each sampler, problem, and measurement class will 
be provided). For example, while the OBABO scheme in the code example does only require step size, temperature, and friction, a Metropolized algorithm 
might also take in the number of integrator steps to generate a new proposal.

Similarly, some problem classes need to be constructed with input arguments. An example is the predefined problem 
`BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D` (let me know if you can come up with a shorter, but not less descriptive name!). This problem 
assumes that a 1-dimensional data set of real numbers was sampled from a Gaussian mixture model with 2 components, where the variances of the Gaussian 
components are known but the means are not. It specifies the posterior density of the unknown means given a Gaussian prior.  
In order to use that problem, simply declaring `BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D testproblem;` would not work, because the constructor 
requires a file name (to read in the data set) as well as a batch size (as it offers the option of using subsampled gradients).  
Instead, assuming the data set was stored in a file named **FILENAME.csv**, one would create the object as 
```C++
string filename = "FILENAME.csv";
int batchsize = 20;
BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D testproblem(filename, batchsize); 
```
Note that the `run_mpi_simulation` routine always takes in the same parameters, independently of the sampler, as it is a member function of the sampler 
interface class (see the previous section). 


**Exercise:** Draw samples on the predefined 2-dimensional double well problem with Gaussian basins, using the SGHMC method. You can use the default 
measurement class again. The sampler parameters given in the code snippet above should work here as well. Plot the results to check whether the 
observables converge properly.


### With custom-built measurement objects
Assume that we are happy with the samplers and problems that are already implemented, but we are interested in a certain observable that is not yet covered 
by any of the implemented measurement classes. In this scenario we will have to create our own measurement class. This comes down to editing the 
**measurements.h** file accordingly. To see what we have to do, we take the default measurement class as a model, which is just one of the child classes 
of the measurement interface class `IMEASUREMENT` that all measurement classes must be derived from. 
```C++
class MEASUREMENT_DEFAULT: public IMEASUREMENT{
/* 
This is a default measurement class. It can be used with any model/problem. 
It stores 3 observables: The first model parameter (i.e. the first position coordinate), 
the kinetic, and the configurational temperature given by Force \cdot Position.
*/

    public:
        
        /* 
	Constructor. 
        Note that it calls the interface class' constructor and passes 3 as input for "no_observables", 
	determining the size of the vector "samples". This is somewhat unsafe, as the number passed has to 
	match the number of entries "samples" is expected to have in the "compute_sample" routine below.
        It needs to be changed in the future.
        */ 
        MEASUREMENT_DEFAULT(const size_t n_dist, const bool t_avg = true)
            : IMEASUREMENT(3, n_dist, t_avg) {};
        
        
        void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, 
			    const std:: vector <double>& forces) override {  

            samples[0] = parameters[0];
            samples[1] = 1./parameters.size() * (velocities[0]*velocities[0] + velocities[1]*velocities[1]);
            samples[2] = - 1./parameters.size() * (parameters[0]*forces[0] + parameters[1]*forces[1]); 
        
        };

};
```
To create our own measurement class, we can simply copy'n paste this default measurement class and make the needed adjustments.  
First, we change the class name from `MEASUREMENT_DEFAULT` to `our_new_measurement`. Note that the name also appears with the constructor. 
Next we inspect the arguments of the constructor. We only need to change the number `3` to the number of observables our new class is supposed to collect. 
Since we want to collect two observables, the number is changed to `2`. This usage of “magic numbers” is a design flaw already addressed by the comment 
above the constructor; we ignore it for now. All other arguments nor anything else in the constructor **must not** be changed.  
The last edit we need to do is the modification of the `compute_sample` routine. The signature of the function needs to remain the same, the function 
body needs to be changed. Since we only collect two observables, the third line in the function body needs to be deleted. Then, the right-hand side of 
the first two lines needs to be changed to correspond to the observables we want to take. Rather than simply taking the first model parameter, we want to 
take its third power. And instead of the kinetic temperature, we want to take the sine of the sum of the first two velocity components (note that this 
means that the velocity needs to have two components, i.e. we could not use this new measurement class on a 1-dimensional problem). With these changes, 
our new class is complete: 
```C++
class our_new_measurement: public IMEASUREMENT{

   public:
        
       our_new_measurement(const size_t n_dist, const bool t_avg = true)
           : IMEASUREMENT(2, n_dist, t_avg) {};
        
        
       void compute_sample(const std:: vector <double>& parameters, const std:: vector <double>& velocities, 
       			   const std:: vector <double>& forces) override  {  

           samples[0] = pow(parameters[0],3);
           samples[1] = sin( velocities[0] + velocities[1] );
  
       };

};
```
**Exercise:** Write a measurement class that collects two observables, namely twice the kinetic temperature given by $2T_{kin}=\frac{2}{N_d}p\cdot p$ and half the configurational temperature $\frac{1}{2}T_{conf}=-\frac{1}{2N_d}q\cdot F$. Pick one of the predefined samplers and run them on the same double well problem as in the previous exercise. Plot the results to check whether the observables converge properly.

### With custom-built problem classes
Assume that we have a new potential energy function $U(q)$ whose associated density $\rho$ we want to sample from. We need to add the new problem class to the header file **problems.h**. To see how to write a problem class, we simply compare two of the predefined problems, the 1-dimensinal harmonic oscillator named `HARMONIC_OSCILLATOR_1D` and the 2-dimensional double well problem named `CURVED_DOUBLE_WELL_2D`: 
```C++
class HARMONIC_OSCILLATOR_1D: public IPROBLEM {
/*
This problem is the 1D harmonic oscillator with spring constant \omega^2.
*/

    private:

        const double omega_sq;      // Spring constant K = \omega^2.


    public:

        // Constructor.
        HARMONIC_OSCILLATOR_1D(const double omega_squared = 25, 
			       const std:: vector <double>& init_params = std:: vector <double> {0}, 
			       const std:: vector <double>& init_velocities = std:: vector <double> {0})
        : IPROBLEM(init_params, init_velocities), omega_sq{omega_squared} {
        };

        void compute_force() override {              
                                                               
            forces[0] = -omega_sq * parameters[0];
            
            return;

        };

};


class CURVED_DOUBLE_WELL_2D: public IPROBLEM {
/*
This problem is a 2-dim double well where the wells are connected via a curved channel.
It is defined by  U(x,y) = a(y-x^2)^2  +  x^2 (b-x)^2  with constants a,b >= 0.
The minima lie at (0,0) and (b,b^2).
*/

    private:

        const double a;
        const double b;
        const double min_two_a = -2*a;

    public:

        // Constructor.
        CURVED_DOUBLE_WELL_2D(const double a=1, const double b=2, 
			      const std:: vector <double>& init_params = std:: vector <double> {0,0}, 
			      const std:: vector <double>& init_velocities = std:: vector <double> {0,0})
        : IPROBLEM(init_params, init_velocities), a{a}, b{b} {
        };

        void compute_force() override {              
                                                               
            double x = parameters[0];       // Current parameters (positions).
            double y = parameters[1];

            double y_min_x2 = y - x*x;      // Some help constants.
            double b_min_x = b-x;


            forces[0] = -2*x * ( min_two_a * y_min_x2  -  x * b_min_x  +  b_min_x * b_min_x );
            forces[1] = min_two_a * y_min_x2;
            
            return;

        };

};

```

The two classes differ at 4 points:  
1. They have different class names (note the class name always appears at two occasions).

2. They have different private member variables. These are the variables used by the force computation routine. They can be variables defining the potential (such as the spring constant for the oscillator), or help variables that make computations more efficient (such as `min_two_a` in the double well case). Note that both problems inherit the member vectors `parameters`, `velocities`, and `forces` from the interface class. 

3. Due to different class names and different member variables, the constructors look differently as well, but the structure is always the same. The first arguments passed to the constructor always correspond to the member variables unique to the particular problem class. They are given default values, eg. 25 for the spring constant of the oscillator. The last two arguments of the constructors are always the parameter and velocity vectors that the samplers are operating on (these member variables are inherited from the parent class `IPROBLEM` and are thus automatically visible in every problem class derived from it). They, too, get default values for initialization. Note, however, that their dimension differs: In the case of the 1-dimensional oscillator they are 1-dimensional vectors, as opposed to the 2-dimensional case of the double well.  
The default values allow for the creation of the objects without passing any parameters, eg.: 
```C++
    HARMONIC_OSCILLATOR_1D testproblem_default;     /* uses spring constant 25
                                                       and initial conditions (q,p)=(0,0). */
    
    HARMONIC_OSCILLATOR_1D testproblem_custom(10);  /* uses spring constant 10
                                                       and initial conditions (q,p)=(0,0). */
```
In the next line of the constructor, the parameters and velocity vectors are passed to the parent class constructor, `IPROBLEM(init_params, init_velocities)`. This part **must remain unchanged**. After this, in the same line, we have the member initialization lists, where the member variables unique to the problem class are initialized via the input arguments passed to the constructor. The constructor ends with an empty function body given by `{}`. For more complex problems, this function body would not be empty. In the case of a data science problem, this could be the place where the data set is read in from a file and stored in a vector (see the predefined problem class `BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D`). Also, when using noise in the force evaluations, this would be the place where the random number generator is created and initialized.

4. Lastly, the functions `compute_force` differ between problems. These functions typically don't take input arguments. Given the member variables defining the potential (`omega_sq` for the oscillator, and `a` and `b` for the double well), and given the current entries in the member vector `parameters` (storing the current “positions” of the model), this function computes the forces $-\nabla U(q)$ and stores the results in the (inherited) member vector `forces`. You may convince yourself that the formulas used to compute the force entries are indeed computing the negative gradient of the corresponding potentials (and let me know if they don't!). Note that `compute_force` does not have to compute $-\nabla U(q)$ exactly. Adding noise terms, for example, is perfectly fine.  

Having understood where the source code of two different problem classes differs and where it is the same, we are in a good position to come up with our own problem classes.  

**Exercise:** The Rosenbrock potential in 4 dimensions is given by $$U(x_{1},x_{2},x_{3},x_{4})=\sum_{i=1}^{3}\Big[100\big(x_{i+1}-x_{i}^{2}\big)^{2}+\big(1-x_{i}\big)^{2}\Big]$$ and has 2 minima, one at (1,1,1,1) and one close to (-1,1,1,1). Write a problem class corresponding to this problem. Then, using one of the predefined samplers and the default measurement class, collect samples from it. Examine the results via plotting.

### With custom-built samplers
Lastly, we need to know how to add a novel sampler to the library. This setting is a bit more complex as the definition of the class itself and their member functions are in different files. In the header **samplers.h** we need to define our new class. Comparing two samplers that are already there, the `OBABO_sampler` and the `SGHMC_sampler`, we see that, apart from the class name, they don't differ at all: 
```C++
class OBABO_sampler: public ISAMPLER{
/*
The OBABO splitting scheme.
*/

    private:

        const double T;      // Temperature.
        const double gamma;  // Friction.
        const double h;      // Stepsize.

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, 
			     const int randomseed, const int t_meas) override; 


    public:

        OBABO_sampler(const double T, const double gamma, const double h): T{T}, gamma{gamma}, h{h} {     // Constructor.
        }; 

        void print_sampler_params() override;  

        ~OBABO_sampler(){};            // Destructor.

};


class SGHMC_sampler: public ISAMPLER{
/*
The SGHMC sampler (Chen et al. 2014).
Note that whether stochastic gradients are actually used depends on the force routine specified in the 
problem class passed as an argument.
*/

    private:

        const double T;
        const double gamma;
        const double h;

        void draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, 
			     const int randomseed, const int t_meas) override;
   
   
    public:

        SGHMC_sampler(double T, double gamma, double h): T{T}, gamma{gamma}, h{h} {        // Constructor.
        };

        void print_sampler_params() override;

        ~SGHMC_sampler(){};              // Destructor

};
``` 
The member variables temperature `T`, friction `gamma`, and step size `h` are used by most samplers and initialized by the corresponding constructors. If we wanted to implement a Metropolized scheme, we would probably have an additional member variable governing the integrator steps taken to propose a new sample.  

Having added the class of our new scheme to the header file, we can move on to define how the scheme is going to collect samples, i.e. we need to write the function `draw_trajectory` in the source file **samplers.cpp**. The case of the `SGHMC_sampler` serves again as an example: 
```C++
void SGHMC_sampler::draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, 
				    const int randomseed, const int t_meas){

    // Integrator constants.
    const double one_minus_hgamma = 1-h*gamma;    
    const double noise_pref = sqrt(2*h*gamma*T);  

    size_t No_params = problem.parameters.size();  // Number of parameters.

    // COMPUTE INITIAL FORCES.
    problem.compute_force();

    // COMPUTE INITIAL MEASUREMENTS.
    RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);

    // PREPARE RNG.
    std:: mt19937 twister;
    
    std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};
    std:: vector < std::uint32_t > seeds(1);
    seq.generate(seeds.begin(), seeds.end());
    twister.seed(seeds.at(0)); 

	std:: normal_distribution<> normal{0,1};
    double Rn;

	auto t1 = std:: chrono::high_resolution_clock::now();

    // MAIN LOOP.
    for ( size_t i = 1;  i <= max_iter;  ++i ) {
        
        // UPDATE.
        for ( size_t j = 0;  j < No_params;  ++j ) {                  
            
	    // Momentum update.
            Rn = normal(twister);
            problem.velocities[j] = one_minus_hgamma * problem.velocities[j]  +  noise_pref * Rn  +  h * problem.forces[j]; 
            
	    // Parameter update.
	    problem.parameters[j] += h * problem.velocities[j];                                                            
        
        }
  
        // COMPUTE NEW FORCES.
        problem.compute_force();
					 

        // TAKE MEASUREMENT.
		if( i % t_meas == 0 ) {                                                 
            		RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);        
		}
		
        if( i % int(1e5) == 0 ) std:: cout << "Iteration " << i << " done!\n";
	
	}  // END MAIN LOOP.


    // FINALIZE.
    auto t2 = std:: chrono:: high_resolution_clock:: now();
    auto ms_int = std:: chrono:: duration_cast < std:: chrono:: seconds > (t2 - t1);
    std:: cout << "Execution took " << ms_int.count() << " seconds!\n";
        
    return;

};
```
Note that the routine operates on the parameters and velocities that are part of the problem class passed as an input argument. Since the sampler itself does not know the dimension of these vectors, it is important to ask for their size at the beginning and loop over them when they are updated.  
In most cases, the samplers will only differ in two points: They will have different help constants used by the integrators (at the very top of the function body in the snippet above). And, of course, the update steps in the main loop will be different. Note that forces are computed via `problem.compute_force();` and measurements need to be taken via `RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);`. Where the measurements are taken **should not be changed from sampler to sampler**.  

A lot of the code, such as the initialization of the random number generator will be present in all of the samplers. There are plans to introduce another layer of abstraction so that the user only has to specify the particular update step for the sampler, i.e. what happens in a single iteration of the main loop, rather than rewriting the whole code snippet above. However, as different samplers might have different loop structures, in particular the Metropolis-adjusted compared to the unadjusted ones, this needs some more consideration.

Lastly, from the class definitions in the header file above, we see that the samplers also have a routine called `print_sampler_params`, which is called by the parent class `ISAMPLER` at the start of the MPI simulation. This function has a default implementation (which does nothing), so it does not have to be implemented when writing a new sampler. However, if we want our new scheme to print out some information at the start of the simulation, we can just add it to the **samplers.cpp** file. The `SGHMC_sampler`, for example, comes with 
```C++
void SGHMC_sampler::print_sampler_params(){
    
    std:: cout << "SGHMC sampling with parameters:\n";
    std:: cout << "Temperature = " << T << ",\nFriction = " << gamma << ",\nStepsize = " << h << ".\n" << std:: endl; 

};
````
**Exercise:** Implement the ABO-splitting scheme for Langevin dynamics as a new sampler class. You can take the `OBABO_sampler` as guidance. Load the 1-dimensional harmonic oscillator as well as the default measurement class. Run the sampler and plot the results to check the convergence of the observables.
