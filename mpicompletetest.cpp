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
#include <queue>
#include <mpi.h>
#include <chrono>

const double k = 8.99e9;
const double CHARGE_PROTON = 1.6e-19;
const double CHARGE_ELECTRON = -1.6e-19;

using namespace std;

struct Particle {
    double x, y;
    char charge;
    Particle(double x, double y, char charge) : x(x), y(y), charge(charge) {}
};

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

struct Task {
    Particle* particle;
    size_t index;
};

// Shared data for worker threads
struct ThreadData {
    KDTree* tree;
    std::queue<Task>* taskQueue;
    std::vector<double>* netForces;
    double cutoffRadius;
    std::mutex* queueMutex;
};

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

// Thread function to process tasks from the queue
void* workerThreadFunction(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // Measure thread processing time
    auto thread_start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        Task task;
        {
            std::lock_guard<std::mutex> lock(*(data->queueMutex));
            if (data->taskQueue->empty()) {
                break;
            }
            task = data->taskQueue->front();
            data->taskQueue->pop();
        }
        std::vector<Particle*> neighbors;
        data->tree->query(task.particle, data->cutoffRadius, neighbors);
        double force = calculateNetForce(task.particle, neighbors);
        (*data->netForces)[task.index] = force;
    }

    auto thread_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> thread_duration = thread_end_time - thread_start_time;
    std::cout << "Thread processing time: " << thread_duration.count() << " seconds" << std::endl;

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
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::string inputFilename = "particles.csv";
    auto start_process_time = std::chrono::high_resolution_clock::now(); // Start process timing
    double cutoffRadius = 6;
    int numThreads = 4;
    if (argc > 1) {
        cutoffRadius = std::stod(argv[1]);
    }
    if (argc > 2) {
        numThreads = std::stoi(argv[2]);
    }

    // Load particles on process 0
    std::vector<Particle*> particles;
    if (world_rank == 0) {
        particles = parseCSV(inputFilename);
    }

    // Broadcast the number of particles to all processes
    size_t numParticles = particles.size();
    MPI_Bcast(&numParticles, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Resize the vector on other processes and broadcast the particles
    if (world_rank != 0) {
        particles.resize(numParticles);
    }
    for (size_t i = 0; i < numParticles; ++i) {
        double x, y;
        char charge;
        if (world_rank == 0) {
            x = particles[i]->x;
            y = particles[i]->y;
            charge = particles[i]->charge;
        }
        MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&charge, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (world_rank != 0) {
            particles[i] = new Particle(x, y, charge);
        }
    }

    // Build the global KDTree with all particles on each process
    KDTree tree(particles);

    // Divide the workload: Each process calculates forces for a subset
    size_t chunkSize = numParticles / world_size;
    size_t startIdx = world_rank * chunkSize;
    size_t endIdx = (world_rank == world_size - 1) ? numParticles : startIdx + chunkSize;

    std::vector<Particle*> localParticles(particles.begin() + startIdx, particles.begin() + endIdx);
    std::vector<double> localNetForces(localParticles.size(), 0.0);
    std::queue<Task> taskQueue;
    for (size_t i = 0; i < localParticles.size(); ++i) {
        taskQueue.push({localParticles[i], i});
    }

    std::mutex queueMutex;
    ThreadData threadData = {&tree, &taskQueue, &localNetForces, cutoffRadius, &queueMutex};

    pthread_t threads[numThreads];
    auto start_thread_time = std::chrono::high_resolution_clock::now(); // Start thread timing
    for (int i = 0; i < numThreads; ++i) {
        pthread_create(&threads[i], nullptr, workerThreadFunction, &threadData);
    }
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    auto end_thread_time = std::chrono::high_resolution_clock::now(); // End thread timing

    std::vector<double> globalNetForces(numParticles, 0.0);
    MPI_Gather(localNetForces.data(), localNetForces.size(), MPI_DOUBLE, 
               globalNetForces.data(), localNetForces.size(), MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

    auto end_process_time = std::chrono::high_resolution_clock::now(); // End process timing

    if (world_rank == 0) {
        std::chrono::duration<double> process_duration = end_process_time - start_process_time;
        std::chrono::duration<double> thread_duration = end_thread_time - start_thread_time;

        writeForcesToCSV("mpi_net_forces.csv", globalNetForces);
        std::cout << "Net forces written to mpi_net_forces.csv" << std::endl;
        std::cout << "Total process time: " << process_duration.count() << " seconds" << std::endl;
        std::cout << "Total thread time: " << thread_duration.count() << " seconds" << std::endl;
    }

    for (Particle* particle : particles) {
        delete particle;
    }

    MPI_Finalize();
    return 0;
}
