



Task 1 Implementation Breakdown

1. Create a function that constructs a neural network with  layout structure (e.g. 1-3-1)
2. Assign random weights to the constructed neural network.
3. Assign random weights to the constructed neural network.

    Need to decide what is the best data structure for this. So far we have agreed on using dictionaries:
    neuralNEt = {
        "L1" : [[w11, w12, w13], [b11, b12, b13], [a11, a12 ,a13]]
        "L2" : [[w21, w22, w23], [b21], [a21]]
    }
    Each key is a layer and the corresponding value is a list of [weights, bias, activation funcitons]

    Step 1-3 constructs the whole neural net.
4. Implement forward pass and get output.
5. Calculate the loss and record it.
6. Implement backwards propagation
7. repeat steps 4 to 6 until a good set of weights are obtained.
8. repeat steps 1 to 7 for different combintations of activation functions to see which one is the best.

Decisions:
Use Python to implement code
Use Github to store and share files
Use Visual Studio Code for liveshare collab sessions (code together at the same time)
Use discord for meetings? 

Next meeting: Thurs 3rd Nov 11am-1pm if everyone is free (in person?)