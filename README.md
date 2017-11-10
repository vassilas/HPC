# Matrix Multiplication in Cuda

This implementation performs the multiplication `A^T*A` of a matrix `A`, in CUDA. This repository consists of three parts. The first one, uses *cublasDgemm*function, to perform the multiplication. The second one, uses global memory and a simple loop construct to perform the multiplication. The third and last one, utilizes multiple optimized constructs to outperform the simple *cublasDgemm*implementation. Finally, all of the above implementations are benchmarked against each other. Check the [report.pdf](./report.pdf) for a detailed walk through of the project.

## Algorithm Design

The third implementation works with a block multiplication paradigm, where only matrix `A` is stored into global memory, and every block of threads maps to a tile to perform the final multiplication.  Using tiles to perform the multiplication, achieves better locality of references achieving higher memory throughput, and faster execution.

In detail, since we run tests on a *Tesla C2075*GPU (compute capability 2), a tile size of `48*48` cells and a block size of `16*16` threads are created. In this fashion, every thread maps to 9 cells, performing 9 multiplication of the final matrix `C`. Each thread utilizes three different sets of registers:
1) The registers `rC[3][3]`, are used to store the partial product, before kernel's termination and final update of matrix `C`.
2) The registers `rA[3]` and `rA_T[3]`, are used to store a column and a row, from a block of matrix `A` respectively. Those cells are loaded from shared memory and are later used to calculate the value of `rC`.
3) The registers `ra[3]` and `ra_T[3]`, are used for prefetching the next tile from the slow global memory to the faster shared memory.

Registers `rC` are zeroed and thread offsets are calculated to initiate the first transfer from global to shared memory. All threads are synced and the main loop starts calculating the matrix product in tiles. Before the actual calculation of this partial product, the prefetching of the next tile is initiated, as described before, storing the new tile data in the `ra`, `ra_T` registers. From the currently available tile in shared memory, registers `rA`, `rA_T` are populated. The result of `rC` is calculated easily as:

```math
rC[n][m] += rA_T[m] * rA[n]
```

When all the threads in a block finish with this calculation, they transfer the new tile parts, from the registers `ra`, `ra_T`, back to shared memory. The final tile is calculated and the final values of `rC` are stored back to global memory. Since the product of `A^T*A` is a symmetric matrix `C`, we only calculate half of the matrix (upper triangular in our case) and copy the results to their symmetric indices in `C`.

## Testing

The three implementations, are benchmarked against their execution time. The graph bellow shows their performance on different matrix sizes of `A`:

<img alt="graph" src="./images/time.png"> 

## Compiling 

To compile each implementation, simply run the `make` command. The available targets are `erwt1`, `erwt2`, `erwt3`. Depending on your GPU, `TILE_WIDTH` and `BLOCK_WIDTH` can be chosen appropriately to fully utilize your computing capability.

## Contributors

| <img alt="memaskal" src="https://avatars3.githubusercontent.com/u/782005?v=4" width="48"> | <img alt="vassilas" src="https://avatars1.githubusercontent.com/u/26332565?v=4" width="48">|
| :--: | :--: |
| [memaskal](https://github.com/memaskal)| [vassilas](https://github.com/vassilas) |

