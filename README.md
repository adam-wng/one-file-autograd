# one-file-autograd
A single file implementing a working autogradient completely in C++ with only standard headers. The autogradient tree supports every theoretically possible graph of a model, including ones with multiple inputs, outputs, and bifircations. Also allows for easy user customization of model using a ```.add()``` function. 

Mostly just for shits and giggles.

## The idea
Each operation is specified as a ```Module``` class. When a module is added to a ```Tree``` object, it gets wrapped in a ```GradBlock``` object and the tree updates the linking information. Finally, ```.compile()``` traverses through the linking information to produce a valid order of execution.

## How to specify a model
One simply creates a ```Tree``` object and the layers they wish to use, then specifies the model using the function ```.add()```. Finally, after running ```.compile()```, the user is free to load in inputs to run. 

Specifically, the ```.add()``` function takes in the module to be appended, the indices of the inputs, and a boolean specifying whether the node computes a loss. 
```
Tree t;
int idx_0 = t.add(&Embd, 0); // Layer that takes in indices and returns their embeddings. 0 specifies that this will be connected to the 0th input of indicies
idx_0 = t.add(&Concat, idx_0, 0); // Concatenates the embedding with the 0th input of matrices
idx_0 = t.add(&Linear_1, idx_0, false); // Applies linear layer

int loss_0 = t.add(&loss_0, idx_0, true); // 0-th loss layer

int idx_1 = t.add(&Embd, 1); // Embeds the 1st input of indices using the same embedding weights as before

array<int> concat_idx(new int [2], 2);
concat_idx.set(idx_0, 0);
concat_idx.set(idx_1, 1);
idx_0 = t.add(&Concat, concat_idx, false); // Concatenates the two inputs

int loss_1 = t.add(&loss_1, idx_0, true); // A different loss layer

array<int> loss_idx(new int [2], 2);
loss_idx.set(loss_0, 0);
loss_idx.set(loss_1, 1);
t.add(&Average, &loss_idx, false);
t.compile(); // Free to run!
```
Besides wacky models like the one above, the implemenation also allows for the usage of for loops to initialize and specify much larger models with repetitve elements. 

Modules supported are embedding, left and right matrix mult, concatenation, residual connections, selu, relu, and cross entropy.

