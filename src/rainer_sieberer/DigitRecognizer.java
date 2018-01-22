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

/**
 * sample code for PS Software Engineering, ws17: starting point for A12
 * (initial python implementation: Make Your Own Neural Network, Tariq Rashid)
 * (this code is based on submissions from C.Moesl, A.Schuetz, T.Hilgart, M.Regirt)
 */
public class DigitRecognizer {

	private NeuralNetwork net;

	private static final int amountOfTargets = 10;

	/**
	 * Loads the .csv file with the training data or throws an Exception if anything goes wrong;
	 * returns true iff the initialization completed successfully.
	 *
	 * @param csvTrainingData
	 *            the data used to train the neural network
	 * @return true if the initialization was successful
	 */
	public boolean init(File csvTrainingData) throws Exception
	{
		net = new NeuralNetwork();
		int[] numberOfNodes = new int[]{784,200,10};
		List<Function<Double, Double>> activation = new ArrayList<Function<Double, Double>>();
		activation.add( (Double x) -> 1.0 / (1.0 + Math.exp(-x)) );
		activation.add( (Double x) -> 1.0 / (1.0 + Math.exp(-x)) );
		return net.init( csvTrainingData, numberOfNodes, 5, 0.1, activation );
	}

	/**
	 * trains the neural network used for digit recogniztion.
	 *
	 * @return true iff the training of the neural network was successful.
	 * @throws Exception
	 */
	public boolean train() throws Exception
	{
		File csvTrainingData = net.getTrainingFile();
		List<List<Double>> trainingData = new ArrayList<>();
		List<Integer> trainingDataSolution = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new FileReader(csvTrainingData))) {
			

			br.lines().forEach(s -> {
				String[] allValues = s.split(",", 2);

				trainingDataSolution.add(Integer.parseInt(allValues[0]));

				// scale and shift the inputs
				trainingData.add(Arrays.stream(allValues[1].split(","))
						.mapToDouble(x -> Double.parseDouble(x) / 255.0 * 0.99 + 0.01)
						.boxed()
						.collect(Collectors.toList()));
			});
		}

		return net.train(trainingData, trainingDataSolution);
	}

	/**
	 * Tries to recognize the digit represented by csvString.
	 *
	 * @param csvString
	 *            the digit pattern as CSV string.
	 * @return the recognized digit
	 */
	public int recognize(String csvString) throws Exception {
		// scale and shift the inputs
		double[] inputs = Arrays.stream(csvString.split(","))
				.mapToDouble(s -> Double.parseDouble(s) / 255.0 * 0.99 + 0.01)
				.toArray();

		double[] outputs = net.query(inputs);

		// the index of the highest value corresponds to the label
		return indexFromMax(outputs);
	}

	private int indexFromMax(double[] data) {
		int max = 0;

		for (int i = 0; i < data.length; i++)
			if (data[i] > data[max])
				max = i;

		return max;
	}

	private double[] toArray(Matrix matrix) {
		double[] result = new double[matrix.getRowDimension()];

		for (int i = 0; i < result.length; i++)
			result[i] = matrix.get(i, 0);

		return result;
	}
}
