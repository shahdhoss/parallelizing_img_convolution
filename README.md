# Parallelizing Image Convolution

This repository contains implementations of **image convolution** using different parallelization techniques to study performance, scalability, and fault tolerance. The project is divided into three main sections:

- MPI
- OpenMP
- Distributed fault tolerant implementation using gRPCs

---

## Project Structure
```
parallelizing_img_convolution/
│
├── mpi/           # MPI-based convolution
├── openmp/        # OpenMP-based convolution
├── distributed/   # Distributed fault tolerant implementation
└── README.md
```

---

## Dependencies

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install build-essential
sudo apt install libopenmpi-dev openmpi-bin
```

---

## MPI Implementation

### Description

The MPI version uses message passing to distribute parts of the image across multiple processes. Each process performs convolution on its assigned chunk, then results are gathered to form the final image. This model is suitable for distributed systems and clusters.

### Build
```bash
mpicxx  mpi/mpi.cpp -o mpi
```

### Run
```bash
mpiexec -n 4 ./mpi 
```

`-n 4` specifies the number of MPI processes.

---

## OpenMP Implementation

### Description

The OpenMP version uses shared-memory parallelism, where convolution loops are parallelized across multiple threads running on a single machine.

### Build
```bash
g++ -fopenmp -o parallel openMP/parallel/parallel.cpp
```

### Run
```bash
export OMP_NUM_THREADS=8
./parallel
```

`OMP_NUM_THREADS` controls the number of threads.

---

## Distributed Fault Tolerant Implementation

### Description



### Build
```bash

```

### Run
```bash

```


---
