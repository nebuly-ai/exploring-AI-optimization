<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> •
  <a href="https://github.com/nebuly-ai/learning-AI-optimization#contribute">Contribute to the library</a>
</p>


<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">

# Compilers

Find below an overview on compilers, while [here]() a curated collection of resources on the topic.

Deep Learning Compilers are custom compilers that are used to compile DeepLearning Models.

To compile means to convert the high level code in low level code containing machine level instructions that can be DIRECTLY used by the given Hardware Platform.

One of the main advantages is trivial: usually deep learning models are python-based meaning that the code needs to run on top of the python interpreter that is flexible but slow. 

Libraries such as Pytorch, Numpy and many others rely on a structure composed by two pieces: 

- A C++ compiled library that comprises all the function that are compute intensive (i.e. the actual algebraic operations).
- A Python libraries that is binded with the C++ library that allow to calls compiled C++ functions.

with this strategy the overhead of the python interpreter can be limited but not eliminated, allowing to merge the efficient C++ compiled code with the flexibility and ease to use of Python.

Moreover it is possible to link compiled CUDA kernels to exploit accelerators such as GPUs. 

Deep Learning Compilers instead take as input your python model and compile it as whole returning an object that can be treated as a function. This enables a compiled model to be generally faster as is, just because it do not rely on the Python Interpreter. 

Furthermore, DeepLearning specialized compilers, comprise a set of optimizations designed to enhance the efficiency of a model that are custom-made for DL. These features will be the focus of the next paragraphs, starting from what are the main components of a deep learning compiler and following with the most common optimizations that are build on top of them. 

The main Deep Learning Compilers are:

- TVM (Apache Open Source)
- Glow / Tensor Comprehension   (Meta)
- XLA (Google)
- ONNX Runtime
- OpenVino (Intel)
- Tensor-RT (NVIDIA)

## Compiler's structure

At a high level, the main building block of a compiler is the IR, which stands for Intermediate Representation. Basically, the IR is a way to abstract the code to be compiled, so that some features of the code can be analyzed and optimized. To give an example, the first intermediate representation of a deeplearning compiler is usually the model graph: representing the model as a graph of algebraic operations allows dependencies between operations, data flow, and sometimes control flow to be highlighted.

The compiler is essentially made up of many overlapping IRs. The IRs are hierarchical, the higher-level IRs usually affect only the model, while the lower-level IRs usually also take into account the hardware platform on which the code is to run. IRs are pipelined: the model is first fed by the higher-level IRs and then is gradually converted to lower-level IRs.

Where are the optimizations performed? Everywhere! Each IR is designed to better highlight some optimization opportunity, and each IR has its own "steps" that modify the current representation of the model, increasing the performance of the original model. To make this concept tangible, let us imagine the graph of our model in the first high-level IR; this graph can be modified by moving or joining nodes, removing unused branches, and so on, all of these graph optimizations are usually steps in the first intermediate representation.

## High-level IRs

High-level IRs must represent:

- tensor computation
- data (input, weights, and intermediate data).

The design challenge of high-level IRs is the abstraction capability of computation and control flow, which can capture and express different DL models. The goal of high-level IRs is to establish the control flow and dependency between operators and data, as well as to provide an interface for graph-level optimizations.

High-level IRs are:

- IRs based on Data Dependence Draphs (DDG): used for generic compilers. Used to perform optimizations such as:
    - elimination of common subexpressions (CSE)
    - elimination of dead code (DCE)
- IR based on directed acyclic graphs (DAG): nodes of a DAG represent atomic DL operators (convolution, pooling, etc.) and edges represent tensors. It is used to analyze the relationships and dependencies among various operators and use them to guide optimizations.
- IR based Let-Binding: is a method to resolve semantic ambiguity by offering let expression. A let node is generated, which points to the operator and the variable in the expression, instead of simply constructing a computational relationship between the variables like a DAG. Among DL compilers, TVM's Relay IR adopts both DAG-based IR and let-binding-based IR.

### Representation of tensors.

Different graph IRs have different ways of representing calculus on tensors.

- Function-based (GLow, XLA, OpenVino): The function-based representation provides only encapsulated operators. Instructions are organized in three levels:
    - the entire program
    - a function
    - the operation.
- Lambda expression (TVM): describes computation through binding and substitution of variables.
- Einstein notation (Glow): indices of temporary variables need not be defined. IR can understand the actual expression based on the occurrence of undefined variables based on Einstein notation. Operators must be associative and commutative, making further parallelization possible.

### Data representation

DL compilers can represent tensor data directly with memory pointers or, more flexibly, with placeholders.

- Placeholder: A placeholder is simply a variable with explicit shape information (size in each dimension), which will be populated with values at a later stage of computation. It allows programmers to describe operations and construct the computation graph without having to indicate the exact data elements. Allows the shape of inputs/outputs and other corresponding intermediate data to be changed using placeholders without changing the definition of the calculation.
- Unknown (dynamic) shape representation: The unknown dimension is usually supported when declaring placeholders. The unknown shape representation is needed to support the dynamic model. Constraint inference and dimension checking should be relaxed and, in addition, an additional mechanism should be implemented to ensure memory validity.
- Data layout: Data layout describes the organization of a tensor in memory and is usually a mapping from logical indices to memory indices. The data layout usually includes the sequence of dimensions. Combining data layout information with operators rather than tensors allows intuitive implementation for some operators and reduces compilation overhead.
- Boundary inference: Boundary inference is applied to determine the limit of iterators. Boundary inference is usually performed recursively or iteratively, based on the computation graph and known placeholders. Once the limit of the root iterator is determined based on the forms of the placeholders, the other iterators can be inferred based on the recursive relations.

### Types of operations.

In addition to the standard algebraic operators in common use, there are others that are implemented in the DL framework:

- Broadcast: Broadcast operators can replicate data and generate new data with compatible form. Without broadcast operators, the shapes of input tensors are more constrained.
- Control flow: Control flow is required when representing complex and flexible models. Models with recurring relationships and data-dependent conditional execution require control flow.
- Derivation: The derivation operator of an Op operator takes as input the output gradients and input data of Op, then computes the gradient of Op. Programmers can use these derivation operators to construct derivatives of custom operators. In particular, DL compilers that cannot support derivative operators fail to provide model training capability.
- Custom operators: Allows programmers to define their own operators for a particular purpose.


## Front-End Optimizations

### **Node-level optimizations:** compute graph nodes are coarse enough to allow optimizations within a single node.

- **Nop Elimination**: removes no-op instructions that take up a small amount of space but do not specify any operations.
- **Zero-Dim-Tensor Elimination**: is responsible for removing unnecessary operations whose inputs are zero-dimensional tensors.

### **Block Level optimizations**

- **Algebraic simplification**: These optimizations consider a sequence of nodes and exploit the commutativity, associativity and distributivity of different types of nodes to simplify the computation. They can also be applied to DL-specific operators (e.g., reshape, transpose, and pooling). Operators can be reordered and sometimes eliminated. Consists of:
    - Algebraic identification
    - Force reduction: by which we can replace more expensive operators with cheaper ones;
    - Constant folding: by which we can replace constant expressions with their values.
    
    The most common cases in which algebraic simplification can be applied are:
    
    - Computational order optimization: optimization finds and removes remodeling/transposition operations based on specific features.
    - Node combination optimization: optimization combines multiple consecutive transposition nodes into one node, removes identity transposition nodes, and optimizes transposition nodes into remodeling nodes when they do not actually move data.
    - ReduceMean node optimization: the optimization performs ReduceMean replacement with the AvgPool node if the input of the reduce operator is 4D with the last two dimensions to be reduced.
- **Operator merging**: It enables better computation sharing, eliminates intermediate allocations, facilitates further optimization by combining nested cycles, and reduces launch and synchronization overhead.
- **Sinking operators**: This optimization sinks operations such as transpositions below operations such as batch normalization, ReLU, sigmoid and channel shuffle. Through this optimization, many similar operations are brought closer to each other, creating more opportunities for algebraic simplification.

### **Dataflow Level Optimizations:**

- **Common subexpression elimination (CSE)**: An expression E is a common subexpression if the value of E has been previously calculated and the value of E should not be changed after the previous calculation. In this case, the value of E is calculated only once and the already calculated value of E can be used to avoid being recompiled at other points.
- **Dead Code Elimination (DCE):** a set of code is dead if the calculated results or side effects are not used. DCE optimization removes dead code. Dead code is usually not caused by programmers, but by other graph optimizations. Other optimizations, such as dead store elimination (DSE), which removes stores in tensors that will never be used, also belong to DCE.
- **Static memory planning**: is performed to reuse memory buffers as much as possible. Typically, there are two approaches:
    - In-Place Memory Sharing: uses the same memory for the input and output of an operation and merely allocates a copy of the memory before the computation is executed.
    - Standard Memory Sharing: reuses memory from previous operations without overlapping them.
- **Layout transformation**: Layout transformation tries to find the best data layouts to store the tensors in the computation graph and then inserts the layout transformation nodes into the graph. Note that the actual transformation is not performed here, but will be performed during evaluation of the computation graph by the compiler backend. Indeed, the performance of the same operation in different data layouts is different, and even the best layouts are different on different hardware. Some DL compilers rely on hardware-specific libraries to achieve higher performance, and the libraries may require certain layouts. Not only does the layout of tensor data have a nontrivial influence on final performance, but transformation operations also have significant overhead. Because they also consume memory and computational resources.

## Low-level IR.

The low-level IR is designed for hardware-specific optimization and code generation on different hardware targets. Therefore, the low-level IR must be fine enough to reflect hardware characteristics and represent hardware-specific optimizations.

- Halide-based IR (TVM, Glow). The fundamental philosophy of Halide is the separation of computation and programming. Rather than directly providing a specific scheme, compilers adopting Halide try several possible programs and choose the best one. The boundaries of memory references and loop nests in Halide are limited to bounded boxes aligned to axes. Therefore, Halide cannot express computation with complicated patterns. Moreover, Halide can easily parameterize these boundaries and expose them to the adjustment mechanism. Halide's original IR must be modified when applied to the backend of DL compilers.
- IR based on polyhedral models (Glow, OpenVino). The polyhedral model is an important technique adopted in DL compilers. It uses linear programming, affine transformations, and other mathematical methods to optimize loop-based codes with a static control flow of bounds and branches. Unlike Halide, the boundaries of memory references and loop nests can be polyhedra of any shape in the polyhedral model. However, such flexibility also prevents integration with tuning mechanisms. Polyhedron-based IR facilitates the application of various polyhedral transformations (e.g., fusion, tiling, sinking, and mapping), including device-dependent and device-independent optimizations.
- Other unique IRs. There are DL compilers that implement low-level custom IRs without using Halide and the polyhedral model. On these custom low-level IRs, they apply hardware-specific optimizations and reductions to the LLVM IR.

The low-level IR adopted by most DL compilers can eventually be lowered to LLVM's IR and benefit from LLVM's mature optimizer and code generator. In addition, LLVM can explicitly design custom instruction sets for specialized accelerators from scratch. However, traditional compilers can generate poor code when passed directly to LLVM IR. To avoid this situation, DL compilers apply two approaches to achieve hardware-dependent optimization:

- perform a target-specific loop transformation in the upper IR of LLVM (IR based on Halide and Polyhedral e.g., Glow, TVM, XLA, OpenVino)
- provide additional information about the hardware target for optimization steps. (e.g., Glow) Most DL compilers apply both approaches, but the emphasis is different. The compilation scheme in DL compilers can be classified mainly into two categories:
    - **just-in-time (JIT):** can generate executable codes on the fly and can optimize codes with better runtime knowledge.
    - **ahead-of-time (AOT):** they generate all executable binaries first and then execute them. Therefore, they have a wider scope in static analysis than JIT compilation. In addition, AOT approaches can be applied to cross-compilers of embedded platforms and allow execution on remote machines and custom accelerators.

## Back-End Optimizations

The backend transforms high-level IR into low-level IR and performs hardware-specific optimizations. It can use generic third-party tool chains, such as LLVM IR or custom kernels, to leverage prior knowledge of DL models and hardware and generate code more efficiently. Such optimizations are:

- **Hardware Intrinsic Mapping**: can transform a certain set of low-level IR instructions into kernels that are already highly optimized on the hardware.
- **Memory Allocation and Retrieval**: The GPU mainly contains a shared memory space (lower access latency with limited memory size) and a local memory space (higher access latency with high capacity). This memory hierarchy requires efficient memory allocation and fetching techniques to improve data locality.
- **Hiding memory latency**: is used in the backend by reordering the execution pipeline. Since most DL compilers support parallelization on CPU and GPU, memory latency hiding can be naturally achieved via hardware (e.g., context warp switching on the GPU). But for TPU-type accelerators with decoupled access and execution (DAE) architecture, the backend must perform fine-grained scheduling and synchronization to achieve correct and efficient codes.
- **Parallelization**: Since modern processors generally support multi-threading and SIMD parallelism, the compiler back-end must exploit parallelism to maximize hardware utilization and achieve high performance.
- **Loop-oriented optimizations:**
    - **Loop Fusion**: is a loop optimization technique that can merge loops with the same boundaries for better data reuse.
    - **Sliding Windows**: its core concept is to compute values when they are needed and store them on the fly to reuse the data until they are no longer needed. Since sliding windows interleaves the computation of two cycles and makes them serial, it is a compromise between parallelism and data reuse.
    - **Tiling**: splits loops into different tiles, so loops are divided into outer loops that iterate across tiles and inner loops that iterate within a tile. This transformation allows for better localization of data within a tile by placing a tile in hardware caches. Because the size of a tile is hardware-specific, many DL compilers determine the tile's pattern and size by autotuning.
    - **Loop reordering**: changes the order of iterations in a nested loop, which can optimize memory access and thus increase spatial locality. It is specific to data layout and hardware characteristics. However, it is not safe to perform loop unrolling when there are dependencies along the iteration order.
    - **Loop Unrolling**: can unroll a specific loop to a fixed number of copies of loop bodies, which allows compilers to apply aggressive instruction-level parallelism. Usually, loop unrolling is applied in combination with loop splitting, which first splits the loop into two nested loops and then completely unrolls the inner loop.


<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">


<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> •
  <a href="https://github.com/nebuly-ai/learning-AI-optimization#contribute">Contribute to the library</a>
</p>
