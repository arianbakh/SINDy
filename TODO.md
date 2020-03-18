# TODO

- DRY: SINDy toolkit
- adaptive lambda
- plot mse vs. complexity in lorentz -> toolkit
- anything other than complexity and mse for cost?
- use a class for data (incl. x, x_dot, adjacency_matrix)
- save each SINDy run output in a different directory named after job type (lorentz, ...) and 
datetime + save library for each run
- shared settings
- _get_theta takes too long, parallelize with numpy operations
- idea: to speed up, select most active node, its neighbors and neighbors of neighbors
- save progress so you can resume

# genetics

- genetics only for power, find best weights by least squares
- mutation: add statement (a new random power is added with a very low chance)
- mutation: modify power (each of the powers with a very low chance are added a random value from [-0.5, 0.5])
- crossover: keep all genes that are present in both parents
- crossover: P is set of genes in parent 1 that are different from genes of parent 2
- crossover: Q is set of genes in parent 2 that are different from genes of parent 1
- crossover: without loss of generality, assume |P| < |Q|
- crossover: choose random number m in range [|P|, |Q|]
- crossover: choose m genes at random from P union Q
- fitness: AIC too complicated, use complexity * MSE
- what about delete statement? seems unnecessary
- don't need to use all nodes all the time

# experiments

- experiment 4: dynamic number of statements, coefficients
- remove chromosome size
- add complexity to fitness
- experiment 5: two variables
- data-driven discovery of complex dynamical networks using intermittent genetic and SINDy
