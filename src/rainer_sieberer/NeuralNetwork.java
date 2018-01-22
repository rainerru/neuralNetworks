package rainer_sieberer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.Collections;

/**
 * this is a adaptation of the available code for the assignment 12
 * (initial python implementation: Make Your Own Neural Network, Tariq Rashid)
 * (this code is based on submissions from C.Moesl, A.Schuetz, T.Hilgart, M.Regirt)
 */
public class NeuralNetwork {

	private File trainingFile;
	private List<Integer> trainingDataSolution;
	private List<List<Double>> trainingData;
	private int numberOfEpochs;
	private double learningRate;
	private List<Function<Double, Double>> activation;
	private Random random = new Random();
	private int[] numberOfNodes;
	private List<Matrix> weights;

	public File getTrainingFile () { return this.trainingFile; }

	public boolean init(
		File trainingFile, int[] numberOfNodes, int numberOfEpochs,
		double learningRate, List<Function<Double, Double>> activation
 	)
	{
		this.trainingFile = trainingFile;
		this.numberOfNodes = numberOfNodes;
		this.numberOfEpochs = numberOfEpochs;
		this.learningRate = learningRate;
		this.activation = activation;
		this.weights = new ArrayList<Matrix>();

		for ( int i = 0; i < this.numberOfNodes.length - 1; i++ )
		{
			this.weights.add( new Matrix( numberOfNodes[i+1], numberOfNodes[i] ) );
			fillWithRandomValues( this.weights.get(i) , 0 , Math.pow(numberOfNodes[i], -0.5) );
		}

		return true;
	}

	/**
	 * allows to change the initial values by inserting another list of weight matrices;
	 * this method should be called after initializing
	 *
	 * @param	weights	the new list of weight matrices
	 */
	public void changeInitialValues ( int index, Matrix newWeights )
	{
		if ( index < 0 || index >= this.weights.size() )
			return;
		this.weights.set( index, newWeights );
	}

	/**
	 * trains the neural network
	 *
	 * @return true iff the training of the neural network was successful.
	 * @throws Exception
	 */
	public boolean train( List<List<Double>> trainingData, List<Integer> trainingDataSolution ) throws Exception 
	{		
		this.trainingData = trainingData;
		this.trainingDataSolution = trainingDataSolution;
		// create the target output values (all 0.01, except the desired label which is 0.99)
		int amountOfTargets = numberOfNodes[ numberOfNodes.length - 1];
		double[] targets = DoubleStream.generate(() -> 0.01).limit(amountOfTargets).toArray();

		for (int epochs = 0; epochs < this.numberOfEpochs; epochs++) {
			for (int i = 0; i < trainingData.size(); i++) {
				targets[trainingDataSolution.get(i)] = 0.99;

				double[] inputs = Arrays.stream(trainingData.get(i).toArray()).mapToDouble(d -> (double) d).toArray();

				train(inputs, targets);

				targets[trainingDataSolution.get(i)] = 0.01;
			}
		}

		return true;
	}

	private void fillWithRandomValues(Matrix matrix, double mean, double variance) {
		for (int row = 0; row < matrix.getRowDimension(); row++)
			for (int col = 0; col < matrix.getColumnDimension(); col++)
				matrix.set(row, col, nextRandom(mean, variance));
	}

	private double nextRandom(double mean, double variance) {
		return mean + random.nextGaussian() * variance;
	}

	/**
	 * Trains the neural network with the given input and target values.
	 *
	 * @param inputsList
	 *            the input values to be used
	 * @param targetsList
	 *            the target values for the given input values
	 */
	public void train(double[] inputsList, double[] targetsList)
	{
		Matrix inputs = toMatrix(inputsList);
		Matrix targets = toMatrix(targetsList);

		List<Matrix> outputs = feedForward( inputs );
		List<Matrix> errors = backPropagate( outputs, targets );

		Matrix current;
		
		for ( int layer = 0; layer < numberOfNodes.length - 1; layer++ )
		{
			current = this.weights.get( layer );
			this.weights.set( layer, current.matrixAddition( errors.get( layer + 1 )
						.multByElement( outputs.get( layer + 1 ) )
						.multByElement( outputs.get( layer + 1  )
						.applyFuntion( activation.get( layer ) ) )
						.matrixMultiplication( outputs.get( layer ).transposeMatrix())
						.scalarMultiplication( learningRate )) );
		}
	}

	/**
	 * Feeds input values forward through all the layer and returns the list of hidden outputs
	 *
	 * @param inputs
	 *            the input values to be used
	 * @return	the list of hidden outputs, created by applying weights and the activation functions
	 */
	public List<Matrix> feedForward ( Matrix inputs )
	{
		Matrix current = inputs;
		List<Matrix> outputs = new ArrayList<Matrix>();
		outputs.add(inputs);
		for ( int layer = 0; layer < numberOfNodes.length - 1; layer++ )
		{
			current = weights.get( layer ).matrixMultiplication( current );
			current = current.applyFuntion( activation.get( layer ) );
			outputs.add( current );
		}
		return outputs;
	}

	/**
	 * Calculates the output error and propagates this error backwards through the layers
	 *
	 * @param outputs
	 *            the hidden outputs calculated by feedind forward some input values
	 * @param targets
	 *            the targets used to determine the output error
	 * @return	the list of hidden errors
	 */
	public List<Matrix> backPropagate ( List<Matrix> outputs, Matrix targets )
	{
		List<Matrix> errors = new ArrayList<Matrix>();
		Matrix current = targets.matrixSubstraction( outputs.get( numberOfNodes.length-1 ) );
		errors.add( current );
		for ( int layer = numberOfNodes.length - 2; layer >= 0; layer-- )
		{
			current = weights.get( layer ).transposeMatrix().matrixMultiplication( current );
			errors.add( current );
		}
		Collections.reverse( errors );
		return errors;
	}

	/**
	 * Queries the output of the neural network for a given input.
	 *
	 * @param inputsList
	 *            the input to query for.
	 * @return the output from the network.
	 */
	public double[] query(double[] inputsList) {
		Matrix inputs = toMatrix(inputsList);
		List<Matrix> outputs = feedForward( inputs );
		return toArray( outputs.get( numberOfNodes.length - 1 ) );
	}

	public void printInfo ()
	{
		System.out.println("net information:");

		System.out.println("training file: " + trainingFile.getName());
		System.out.println("numberOfEpochs = " + numberOfEpochs);
		System.out.println("learningRate = " + learningRate);

		for ( int i = 0; i < numberOfNodes.length; i++ )
		{
			System.out.println("layer " + i + ": nodes = " + numberOfNodes[i]);
		}
	}

	private Matrix toMatrix(double[] data) {
		Matrix result = new Matrix(data.length, 1);

		for (int i = 0; i < data.length; i++)
			result.set(i, 0, data[i]);

		return result;
	}

	private double[] toArray(Matrix matrix) {
		double[] result = new double[matrix.getRowDimension()];

		for (int i = 0; i < result.length; i++)
			result[i] = matrix.get(i, 0);

		return result;
	}
}
