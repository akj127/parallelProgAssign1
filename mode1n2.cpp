#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <pthread.h>
#include <thread>
#include <mutex>
#include <chrono>

const double k = 8.99e9; // Coulomb's constant
const double CHARGE_PROTON = 1.6e-19; // Charge of a proton
const double CHARGE_ELECTRON = -1.6e-19; // Charge of an electron

using namespace std;

struct Particle {
    double x, y;
    char charge;

    Particle(double x, double y, char charge) : x(x), y(y), charge(charge) {}
};

// KDTree structure (same as before)
struct KDTreeNode {
    Particle* particle;
    KDTreeNode* left;
    KDTreeNode* right;

    KDTreeNode(Particle* particle) : particle(particle), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
    KDTree(const std::vector<Particle*>& particles) {
        std::vector<Particle*> particles_copy = particles;
        root = buildTree(particles_copy, 0);
    }

    void query(Particle* target, double radius, std::vector<Particle*>& results) {
        queryRec(root, target, radius, 0, results);
    }

private:
    KDTreeNode* root;

    KDTreeNode* buildTree(std::vector<Particle*>& particles, int depth) {
        if (particles.empty()) return nullptr;

        int axis = depth % 2;
        std::sort(particles.begin(), particles.end(), [axis](Particle* a, Particle* b) {
            return axis == 0 ? a->x < b->x : a->y < b->y;
        });

        int median = particles.size() / 2;
        KDTreeNode* node = new KDTreeNode(particles[median]);

        std::vector<Particle*> leftParticles(particles.begin(), particles.begin() + median);
        std::vector<Particle*> rightParticles(particles.begin() + median + 1, particles.end());

        node->left = buildTree(leftParticles, depth + 1);
        node->right = buildTree(rightParticles, depth + 1);

        return node;
    }

    void queryRec(KDTreeNode* node, Particle* target, double radius, int depth, std::vector<Particle*>& results) {
        if (!node) return;

        double dist = std::sqrt(std::pow(node->particle->x - target->x, 2) + std::pow(node->particle->y - target->y, 2));
        if (dist < radius && node->particle != target) {
            results.push_back(node->particle);
        }

        int axis = depth % 2;
        double delta = axis == 0 ? target->x - node->particle->x : target->y - node->particle->y;

        if (delta < 0) {
            queryRec(node->left, target, radius, depth + 1, results);
            if (std::abs(delta) < radius) {
                queryRec(node->right, target, radius, depth + 1, results);
            }
        } else {
            queryRec(node->right, target, radius, depth + 1, results);
            if (std::abs(delta) < radius) {
                queryRec(node->left, target, radius, depth + 1, results);
            }
        }
    }
};

// Shared data for pthread
struct ThreadData {
    int thread_id;
    int num_threads;
    const std::vector<Particle*>* particles;
    KDTree* tree;
    std::vector<double>* netForces;
    double cutoffRadius;
    vector<double>* threadTimes;
};

// Mutex for thread-safe output
std::mutex force_mutex;

// Function to calculate the net force on a particle
double calculateNetForce(Particle* target, const std::vector<Particle*>& neighbors) {
    double netForce = 0.0;
    for (Particle* neighbor : neighbors) {
        double distance = std::sqrt(std::pow(neighbor->x - target->x, 2) + std::pow(neighbor->y - target->y, 2)) * 1e-10;
        double force = k * ((target->charge == 'p' ? CHARGE_PROTON : CHARGE_ELECTRON) *
                            (neighbor->charge == 'p' ? CHARGE_PROTON : CHARGE_ELECTRON)) / (distance * distance);
        netForce += force;
    }
    return netForce;
}

// Pthread function for parallel force calculation
void* threadFunction(void* arg) {
    auto start_time = std::chrono::high_resolution_clock::now();
    ThreadData* data = (ThreadData*)arg;
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;
    const std::vector<Particle*>* particles = data->particles;
    KDTree* tree = data->tree;
    std::vector<double>* netForces = data->netForces;
    double cutoffRadius = data->cutoffRadius;
    std::vector<double>* threadTimes = data->threadTimes;

    size_t particles_per_thread = particles->size() / num_threads;
    size_t start_idx = thread_id * particles_per_thread;
    size_t end_idx = (thread_id == num_threads - 1) ? particles->size() : start_idx + particles_per_thread;

    for (size_t i = start_idx; i < end_idx; ++i) {
        std::vector<Particle*> neighbors;
        tree->query((*particles)[i], cutoffRadius, neighbors);
        double force = calculateNetForce((*particles)[i], neighbors);

        std::lock_guard<std::mutex> lock(force_mutex);
        (*netForces)[i] = force;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> thread_duration = end_time - start_time;
    (*threadTimes)[thread_id] = thread_duration.count();
    return nullptr;
}

// Function to load particles from a CSV file
std::vector<Particle*> parseCSV(const std::string& filename) {
    std::vector<Particle*> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token1, token2, token3;
        if (std::getline(ss, token1, ',') && std::getline(ss, token2, ',') && std::getline(ss, token3, ',')) {
            try {
                double x = std::stod(token1);
                double y = std::stod(token2);
                char charge = token3[0];
                data.push_back(new Particle(x, y, charge));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error parsing line: " << line << std::endl;
            }
        } else {
            std::cerr << "Invalid line format: " << line << std::endl;
        }
    }
    file.close();
    return data;
}

// Function to write net forces to a CSV file
void writeForcesToCSV(const std::string& filename, const std::vector<double>& netForces) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (const double& force : netForces) {
        file << force << std::endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    const std::string inputFilename = "particles.csv";
    double cutoffRadius = 6;
    std::string mode = "sequential";
    int num_threads = std::thread::hardware_concurrency();

    if (argc > 1) {
        try {
            cutoffRadius = std::stod(argv[1]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid cutoff radius argument. Using default value: " << cutoffRadius << std::endl;
        }
    }

    if (argc > 2) {
        mode = argv[2];
    }
    if (argc > 3) {
        try {
            num_threads = std::stoi(argv[3]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number of threads argument. Using default value: " << num_threads << std::endl;
        }
    }

    const std::string outputFilename = "net_forces_" + std::to_string(cutoffRadius) + "_" + mode + std::to_string(num_threads)+ ".csv";
    std::vector<Particle*> particles = parseCSV(inputFilename);
    std::cout << "Loading done, particle count: " << particles.size() << std::endl;

    KDTree tree(particles);
    std::vector<double> netForces(particles.size(), 0.0);

    if (mode == "sequential") {
        std::cout << "Running in sequential mode..." << std::endl;
        auto start_serial = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < particles.size(); ++i) {
            std::vector<Particle*> neighbors;
            tree.query(particles[i], cutoffRadius, neighbors);
            netForces[i] = calculateNetForce(particles[i], neighbors);
        }
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_serial = end_serial - start_serial;
        std::cout << "Sequential duration: " << duration_serial.count() << " seconds" << std::endl;
    }
    else if (mode == "pthread") {
        
        std::cout << "Running in pthread mode with " << num_threads << " threads..." << std::endl;

        pthread_t threads[num_threads];
        ThreadData threadData[num_threads];
        std::vector<double> threadTimes(num_threads, 0.0); 

        auto start_thread_creation = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_threads; ++i) {
            threadData[i] = {i, num_threads, &particles, &tree, &netForces, cutoffRadius, &threadTimes};
            pthread_create(&threads[i], nullptr, threadFunction, &threadData[i]);
        }
        auto end_thread_creation = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], nullptr);
        }
        auto end_parallel_thread_creation = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> parallel_time = end_parallel_thread_creation - end_thread_creation;
        std::chrono::duration<double> thread_creation_time = end_thread_creation - start_thread_creation;
        double total_thread_exec_time = 0.0;
        for (unsigned int i = 0; i < num_threads; ++i) {
            total_thread_exec_time += threadTimes[i];
        }

        // Print or process the thread times
        for (int i = 0; i < num_threads; ++i) {
            cout << "Thread " << i << " completed in " << threadTimes[i] << " seconds" << std::endl;
        }
        std::cout << "Thread Creation Time: " << thread_creation_time.count() << " seconds\n";
        std::cout << "Total Parallel Duration (including thread creation): " << parallel_time.count() + thread_creation_time.count() << " seconds\n";
        std::cout << "Individual Thread Execution Times:\n";
        std::cout << "Total Threads Execution Time: " << total_thread_exec_time << " seconds\n\n";
        } else {
            std::cerr << "Invalid mode! Use 'sequential' or 'pthread'." << std::endl;
            return 1;
        }

    writeForcesToCSV(outputFilename, netForces);
    std::cout << "Net forces written to " << outputFilename << std::endl;

    for (Particle* particle : particles) {
        delete particle;
    }

    return 0;
}
