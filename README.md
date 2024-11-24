# Assignment 1: ECE1747H

## Author
**Akhil Jarodia**  
Date: October 27, 2024

### Prerequisites
1. Ensure you have the following software installed:
   - GCC (GNU Compiler Collection)
   - MPI library (e.g., OpenMPI)
   - Pthread library
   - Python (for testing accuracy)

### Compilation
1. Open a terminal and navigate to the directory containing your source code files

2. Compile the sequential and pthread version:
   ```bash
   g++ -std=c++11 -O2 -o -Wall -o mode1n2 mode1n2.cpp 

3. Execute the code using the following command
    ```bash
    ./mode1n2 <cutoff> <pthread or sequential> <Num of threads(optional)> 
4. To compile MPI mode 
    ```bash
     mpic++ -o mpicompletetest  mpicompletetest.cpp -lpthread
5. To run the MPI mode
    ```bash
    mpirun -np <num of process> ./mpicompletetest <cutoff> <num of thread per leader>
 
### Testing
1. Use the following command to compare the csv generated from my code with the oracle solution
    ``` bash
    python3 error_checker.py oracle.csv <your.csv> 

### Graph generation
1. The bar graphs were written in latex along with the report. I'll attach the tex file for your reference which contains my graph code.
